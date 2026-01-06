from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
import logging
import asyncio

_LOGGER = logging.getLogger(__name__)

class InverterDriver(ABC):
    """Abstract base class for inverter drivers."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the driver."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the driver."""
        pass

    @abstractmethod
    def get_required_entities(self) -> List[Dict[str, Any]]:
        """
        Return a list of required entities and their metadata.
        Format: [{"key": "internal_key", "label": "User Label", "domain": "number"}]
        """
        pass
    
    @abstractmethod
    def get_config_schema(self) -> List[Dict[str, Any]]:
        """
        Return a list of configuration parameters.
        Format: [{"key": "max_power_kw", "label": "Max Power (kW)", "type": "float", "default": 9.6}]
        """
        pass

    @abstractmethod
    async def apply_control(
        self, 
        ha_client, 
        entity_map: Dict[str, str], 
        config: Dict[str, Any],
        battery_flows: Optional[Dict[str, float]]
    ) -> None:
        """
        Apply control logic to the inverter.
        
        Args:
            ha_client: HomeAssistant client instance.
            entity_map: Mapping of internal keys to HA entity IDs.
            config: Driver-specific configuration values.
            battery_flows: Dictionary containing battery flow data from optimization plan.
                Keys: grid_to_batt_kw, pv_to_batt_kw, batt_to_load_kw, batt_to_grid_kw, load_kw
                If None, reset inverter to idle/default state.
        """
        pass

class GoodWeDriver(InverterDriver):
    """Driver for GoodWe inverters via Home Assistant integration."""
    
    @property
    def id(self) -> str:
        return "goodwe"

    @property
    def name(self) -> str:
        return "GoodWe Inverter (Experimental)"

    def get_required_entities(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "house_consumption",
                "label": "House consumption sensor",
                "domain": "sensor",
                "default": "sensor.house_consumption"
            },
            {
                "key": "pv_power",
                "label": "PV production sensor",
                "domain": "sensor",
                "default": "sensor.pv_power"
            },
            {
                "key": "soc_sensor",
                "label": "Battery SoC sensor",
                "domain": "sensor",
                "default": "sensor.battery_state_of_charge"
            },
            {
                "key": "operation_mode", 
                "label": "Operation mode select", 
                "domain": "select",
                "default": "select.goodwe_inverter_operation_mode"
            },
            {
                "key": "eco_mode_power", 
                "label": "Eco mode power %", 
                "domain": "number",
                "default": "number.goodwe_eco_mode_power"
            },
            {
                "key": "eco_mode_soc", 
                "label": "Eco mode SoC limit %", 
                "domain": "number",
                "default": "number.goodwe_eco_mode_soc"
            },
            {
                "key": "active_power", 
                "label": "Grid active power sensor", 
                "domain": "sensor",
                "default": "sensor.active_power"
            },
        ]

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "max_power_kw", "label": "Max inverter power (kW)", "type": "float", "default": 9.6},
        ]

    async def apply_control(
        self, 
        ha_client, 
        entity_map: Dict[str, str], 
        config: Dict[str, Any],
        battery_flows: Optional[Dict[str, float]]
    ) -> None:
        mode_entity = entity_map.get("operation_mode")
        power_entity = entity_map.get("eco_mode_power")
        soc_entity = entity_map.get("eco_mode_soc")
        
        _LOGGER.info("GoodWe entity mapping: mode=%s, power=%s, soc=%s", 
                     mode_entity, power_entity, soc_entity)
        
        if not mode_entity or not power_entity or not soc_entity:
            _LOGGER.warning("GoodWe driver missing required entity mappings")
            return

        max_power = float(config.get("max_power_kw", 9.6))
        if max_power <= 0:
            max_power = 9.6
        
        # If battery_flows is None, we're disabling - reset to General mode
        if battery_flows is None:
            _LOGGER.info("Resetting GoodWe to General mode (control disabled)")
            await ha_client.call_service(
                "select", "select_option",
                {"option": "general"},
                target={"entity_id": [mode_entity]}
            )
            return
            
        # Extract battery flow data
        grid_to_batt = battery_flows.get("grid_to_batt_kw", 0.0)
        pv_to_batt = battery_flows.get("pv_to_batt_kw", 0.0)
        batt_to_load = battery_flows.get("batt_to_load_kw", 0.0)
        batt_to_grid = battery_flows.get("batt_to_grid_kw", 0.0)
        load_kw = battery_flows.get("load_kw", 0.0)
        
        # Thresholds
        MIN_POWER_KW = 0.02  # 20W minimum threshold
        
        # Determine intervention type based on battery flows
        target_mode = None
        # Default: disable battery (Idle)
        intervention_type = "DISABLE_BATTERY"
        target_mode = "eco_charge"
        target_soc = 0
        target_power_val = 100  # Max power to ensure mode is active (prevent self-use fallback)

        if grid_to_batt > MIN_POWER_KW:
            # Battery is being charged from grid
            intervention_type = "CHARGE_FROM_GRID"
            target_mode = "eco_charge"
            target_soc = 100  # Charge until full
            target_power_pct = min(100, max(0, int((grid_to_batt / max_power) * 100)))
            target_power_val = target_power_pct
            
        elif batt_to_grid > MIN_POWER_KW:
            # Battery is discharging to grid
            intervention_type = "DISCHARGE_TO_GRID"
            target_mode = "eco_discharge"
            target_soc = 0  # Discharge until empty
            discharge_power = batt_to_load + batt_to_grid
            target_power_pct = min(100, max(0, int((discharge_power / max_power) * 100)))
            target_power_val = target_power_pct
            
        elif batt_to_load > MIN_POWER_KW:
            # Battery is covering load (self-consumption)
            intervention_type = "COVER_LOAD"
            target_mode = "general"  # Let inverter manage naturally
            # target_soc and target_power_val remain at defaults (unused for general mode usually, but safe)

        # Execute commands based on mode
        _LOGGER.info(
            "GoodWe Intervention: %s (grid_to_batt=%.2f, batt_to_load=%.2f, batt_to_grid=%.2f)",
            intervention_type, grid_to_batt, batt_to_load, batt_to_grid
        )
        
        if target_mode == "eco_charge" or target_mode == "eco_discharge":
            # Set power percentage first
            if target_power_val is not None:
                await ha_client.call_service(
                    "number", "set_value",
                    {"value": target_power_val},
                    target={"entity_id": [power_entity]}
                )
            
            # Set SoC limit
            if target_soc is not None:
                await ha_client.call_service(
                    "number", "set_value",
                    {"value": target_soc},
                    target={"entity_id": [soc_entity]}
                )
            
            # Set mode last
            resp = await ha_client.call_service(
                "select", "select_option",
                {"option": target_mode},
                target={"entity_id": [mode_entity]}
            )
            _LOGGER.debug("Set mode response: %s", resp)
            
            _LOGGER.info(
                "GoodWe Control Applied: Mode=%s, Power=%d%%, SoC=%d%%",
                target_mode, target_power_val or 0, target_soc or 0
            )
        else:
            # Just set to General mode
            resp = await ha_client.call_service(
                "select", "select_option",
                {"option": target_mode},
                target={"entity_id": [mode_entity]}
            )
            _LOGGER.debug("Set mode response: %s", resp)
            
            _LOGGER.info("GoodWe Control Applied: Mode=%s", target_mode)


class InverterManager:
    def __init__(self):
        self._drivers: Dict[str, InverterDriver] = {}
        self._active_driver_id: Optional[str] = None
        self._entity_map: Dict[str, str] = {}
        self._driver_config: Dict[str, Any] = {}
        
        # Register default drivers
        self.register_driver(GoodWeDriver())

    def register_driver(self, driver: InverterDriver):
        self._drivers[driver.id] = driver

    def get_drivers(self) -> List[InverterDriver]:
        return list(self._drivers.values())

    def get_driver(self, driver_id: str) -> Optional[InverterDriver]:
        return self._drivers.get(driver_id)

    def set_active_driver(self, driver_id: str, entity_map: Dict[str, str], config: Dict[str, Any]):
        if driver_id and driver_id not in self._drivers:
            raise ValueError(f"Unknown driver: {driver_id}")
        self._active_driver_id = driver_id
        self._entity_map = entity_map
        self._driver_config = config

    def get_config(self) -> Dict[str, Any]:
        return {
            "driver_id": self._active_driver_id,
            "entity_map": self._entity_map,
            "config": self._driver_config
        }
    
    def get_active_driver_id(self) -> Optional[str]:
        """Get the currently active driver ID."""
        return self._active_driver_id


    async def apply_control(self, ha_client, battery_flows: Optional[Dict[str, float]]) -> None:
        if not self._active_driver_id:
            _LOGGER.warning("Control skipped: No inverter driver configured. Please configure a driver in Settings > Inverter Control.")
            return

        driver = self._drivers.get(self._active_driver_id)
        if not driver:
            _LOGGER.error("Control skipped: Driver '%s' not found in registry", self._active_driver_id)
            return

        if battery_flows:
            _LOGGER.info("Applying control via %s with battery flows", driver.name)
        else:
            _LOGGER.info("Resetting %s (control disabled)", driver.name)
        
        await driver.apply_control(
            ha_client,
            self._entity_map,
            self._driver_config,
            battery_flows
        )
