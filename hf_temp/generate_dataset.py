"""
🔥 FORGE INTELLIGENCE - SYNTHETIC DATASET GENERATOR
Generates realistic foundry sensor data for ML training and AI assistant
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'output_dir': 'static/data',
    'num_records': 10000,
    'start_date': '2024-01-01',
    'interval_minutes': 5,
    
    # Temperature parameters (°C)
    'temp_base': 1420,
    'temp_min': 1350,
    'temp_max': 1550,
    'temp_optimal_low': 1410,
    'temp_optimal_high': 1430,
    
    # Energy parameters (kWh)
    'energy_base': 450,
    'energy_min': 350,
    'energy_max': 550,
    
    # Anomaly rate
    'anomaly_rate': 0.03,  # 3% anomalies
}

print("╔" + "═"*60 + "╗")
print("║  🔥 FORGE INTELLIGENCE - DATASET GENERATOR              ║")
print("╚" + "═"*60 + "╝")

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================================================
# 1. MAIN SENSOR FUSION DATASET
# ============================================================================

def generate_sensor_fusion_data():
    """Generate comprehensive sensor fusion dataset"""
    print("\n📊 Generating Sensor Fusion Dataset...")
    
    n = CONFIG['num_records']
    timestamps = pd.date_range(
        start=CONFIG['start_date'], 
        periods=n, 
        freq=f"{CONFIG['interval_minutes']}min"
    )
    
    # Time-based patterns
    t = np.arange(n)
    hour_of_day = (t * CONFIG['interval_minutes'] / 60) % 24
    day_of_week = (t * CONFIG['interval_minutes'] / 60 / 24) % 7
    
    # Temperature with realistic patterns
    temp_daily_cycle = 15 * np.sin(2 * np.pi * hour_of_day / 24)  # Daily cycle
    temp_weekly_cycle = 5 * np.sin(2 * np.pi * day_of_week / 7)   # Weekly cycle
    temp_trend = 0.001 * t  # Slight upward trend
    temp_noise = np.random.normal(0, 3, n)
    
    temperature = (CONFIG['temp_base'] + temp_daily_cycle + 
                   temp_weekly_cycle + temp_trend + temp_noise)
    
    # Add anomalies
    anomaly_mask = np.random.random(n) < CONFIG['anomaly_rate']
    anomaly_values = np.where(
        np.random.random(n) < 0.5,
        np.random.uniform(-80, -40, n),  # Cold anomalies
        np.random.uniform(40, 80, n)      # Hot anomalies
    )
    temperature[anomaly_mask] += anomaly_values[anomaly_mask]
    temperature = np.clip(temperature, CONFIG['temp_min'], CONFIG['temp_max'])
    
    # Energy consumption (correlated with temperature)
    energy_base = CONFIG['energy_base']
    temp_deviation = np.abs(temperature - CONFIG['temp_base'])
    energy = energy_base + 0.5 * temp_deviation + np.random.normal(0, 5, n)
    energy = np.clip(energy, CONFIG['energy_min'], CONFIG['energy_max'])
    
    # Multiple sensors
    sensor_ids = np.random.randint(1, 9, n)  # 8 sensors
    
    # Pressure (correlated with temperature)
    pressure = 1.5 + 0.001 * (temperature - 1400) + np.random.normal(0, 0.05, n)
    
    # Flow rate
    flow_rate = 120 + 0.1 * (temperature - 1400) + np.random.normal(0, 3, n)
    
    # Oxygen level
    oxygen = 21 - 0.005 * (temperature - 1400) + np.random.normal(0, 0.3, n)
    oxygen = np.clip(oxygen, 15, 23)
    
    # Carbon content
    carbon = 4.2 + 0.001 * (temperature - 1400) + np.random.normal(0, 0.1, n)
    carbon = np.clip(carbon, 3.5, 5.0)
    
    # Slag viscosity
    slag_viscosity = 2.5 - 0.002 * (temperature - 1400) + np.random.normal(0, 0.2, n)
    slag_viscosity = np.clip(slag_viscosity, 1.0, 4.0)
    
    # Pour quality score (0-100)
    optimal_temp = (CONFIG['temp_optimal_low'] + CONFIG['temp_optimal_high']) / 2
    temp_quality = 100 - np.abs(temperature - optimal_temp) * 2
    energy_quality = 100 - np.abs(energy - energy_base) * 0.5
    pour_quality = (0.6 * temp_quality + 0.4 * energy_quality + np.random.normal(0, 3, n))
    pour_quality = np.clip(pour_quality, 0, 100)
    
    # Anomaly flags
    anomaly_flag = anomaly_mask.astype(int)
    
    # Risk score (0-1)
    risk_score = (np.abs(temperature - optimal_temp) / 100 + 
                  anomaly_flag * 0.3 + 
                  np.random.uniform(0, 0.1, n))
    risk_score = np.clip(risk_score, 0, 1)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': np.round(temperature, 2),
        'energy_kwh': np.round(energy, 2),
        'sensor_id': sensor_ids,
        'pressure_bar': np.round(pressure, 3),
        'flow_rate_lpm': np.round(flow_rate, 2),
        'oxygen_pct': np.round(oxygen, 2),
        'carbon_pct': np.round(carbon, 3),
        'slag_viscosity': np.round(slag_viscosity, 3),
        'pour_quality': np.round(pour_quality, 1),
        'anomaly_flag': anomaly_flag,
        'risk_score': np.round(risk_score, 4),
        'shift': np.where(hour_of_day < 8, 'night', 
                         np.where(hour_of_day < 16, 'day', 'evening')),
        'day_of_week': [['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][int(d) % 7] 
                       for d in day_of_week]
    })
    
    filepath = f"{CONFIG['output_dir']}/sensor_fusion_complete.csv"
    df.to_csv(filepath, index=False)
    print(f"   ✅ Saved: {filepath} ({len(df)} records)")
    return df


# ============================================================================
# 2. HISTORICAL POURING EVENTS DATASET
# ============================================================================

def generate_pouring_events():
    """Generate historical pouring events with outcomes"""
    print("\n🔥 Generating Pouring Events Dataset...")
    
    n_events = 2000
    timestamps = pd.date_range(
        start=CONFIG['start_date'], 
        periods=n_events, 
        freq='2H'  # Pouring every 2 hours on average
    )
    
    # Pre-pour conditions
    pre_temp = np.random.normal(1420, 15, n_events)
    pre_energy = np.random.normal(450, 20, n_events)
    pre_risk = np.random.uniform(0, 0.5, n_events)
    
    # Determine success based on conditions
    optimal_temp = (CONFIG['temp_optimal_low'] + CONFIG['temp_optimal_high']) / 2
    temp_ok = (pre_temp >= CONFIG['temp_optimal_low']) & (pre_temp <= CONFIG['temp_optimal_high'])
    risk_ok = pre_risk < 0.3
    
    success_prob = 0.95 * temp_ok.astype(float) + 0.8 * risk_ok.astype(float)
    success_prob = np.clip(success_prob / 2 + np.random.uniform(-0.1, 0.1, n_events), 0, 1)
    success = np.random.random(n_events) < success_prob
    
    # Pour duration (minutes)
    pour_duration = np.where(success, 
                             np.random.normal(45, 5, n_events),
                             np.random.normal(60, 10, n_events))
    pour_duration = np.clip(pour_duration, 20, 90)
    
    # Yield percentage
    yield_pct = np.where(success,
                         np.random.normal(95, 2, n_events),
                         np.random.normal(85, 5, n_events))
    yield_pct = np.clip(yield_pct, 70, 100)
    
    # Defect rate
    defect_rate = np.where(success,
                           np.random.uniform(0, 2, n_events),
                           np.random.uniform(3, 10, n_events))
    
    # Energy used during pour
    energy_used = pour_duration * 2.5 + np.random.normal(0, 10, n_events)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'pour_id': [f'POUR-{i:05d}' for i in range(n_events)],
        'pre_temperature': np.round(pre_temp, 2),
        'pre_energy_kwh': np.round(pre_energy, 2),
        'pre_risk_score': np.round(pre_risk, 4),
        'pour_duration_min': np.round(pour_duration, 1),
        'yield_pct': np.round(yield_pct, 2),
        'defect_rate_pct': np.round(defect_rate, 2),
        'energy_consumed_kwh': np.round(energy_used, 2),
        'success': success.astype(int),
        'operator_id': np.random.randint(1, 20, n_events),
        'furnace_id': np.random.choice(['F1', 'F2', 'F3'], n_events),
        'iron_grade': np.random.choice(['A', 'B', 'C'], n_events, p=[0.6, 0.3, 0.1])
    })
    
    filepath = f"{CONFIG['output_dir']}/pouring_events.csv"
    df.to_csv(filepath, index=False)
    print(f"   ✅ Saved: {filepath} ({len(df)} records)")
    return df


# ============================================================================
# 3. ALERT HISTORY DATASET
# ============================================================================

def generate_alert_history():
    """Generate historical alerts and incidents"""
    print("\n🚨 Generating Alert History Dataset...")
    
    n_alerts = 500
    timestamps = pd.date_range(
        start=CONFIG['start_date'], 
        periods=n_alerts, 
        freq='8H'
    )
    
    alert_types = ['TEMP_HIGH', 'TEMP_LOW', 'ENERGY_SPIKE', 'ANOMALY_DETECTED', 
                   'SENSOR_FAULT', 'PRESSURE_WARNING', 'FLOW_RATE_LOW', 'MAINTENANCE_DUE']
    alert_weights = [0.25, 0.15, 0.15, 0.2, 0.1, 0.05, 0.05, 0.05]
    
    severity_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    alert_type = np.random.choice(alert_types, n_alerts, p=alert_weights)
    
    # Severity based on alert type
    severity = []
    for at in alert_type:
        if at in ['ANOMALY_DETECTED', 'SENSOR_FAULT']:
            severity.append(np.random.choice(severity_levels, p=[0.1, 0.3, 0.4, 0.2]))
        elif at in ['TEMP_HIGH', 'PRESSURE_WARNING']:
            severity.append(np.random.choice(severity_levels, p=[0.2, 0.4, 0.3, 0.1]))
        else:
            severity.append(np.random.choice(severity_levels, p=[0.4, 0.4, 0.15, 0.05]))
    
    # Response time (minutes)
    response_time = np.random.exponential(15, n_alerts)
    response_time = np.clip(response_time, 1, 120)
    
    # Resolution status
    resolved = np.random.random(n_alerts) < 0.92
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'alert_id': [f'ALT-{i:05d}' for i in range(n_alerts)],
        'alert_type': alert_type,
        'severity': severity,
        'sensor_id': np.random.randint(1, 9, n_alerts),
        'temperature_at_alert': np.round(np.random.normal(1420, 30, n_alerts), 2),
        'response_time_min': np.round(response_time, 1),
        'resolved': resolved.astype(int),
        'resolution_action': np.where(resolved, 
                                      np.random.choice(['AUTO_CORRECTED', 'MANUAL_INTERVENTION', 
                                                       'PARAMETER_ADJUSTED', 'SENSOR_RECALIBRATED'], n_alerts),
                                      'PENDING'),
        'shift': np.random.choice(['day', 'evening', 'night'], n_alerts)
    })
    
    filepath = f"{CONFIG['output_dir']}/alert_history.csv"
    df.to_csv(filepath, index=False)
    print(f"   ✅ Saved: {filepath} ({len(df)} records)")
    return df


# ============================================================================
# 4. ENERGY OPTIMIZATION DATASET
# ============================================================================

def generate_energy_optimization():
    """Generate energy optimization training data"""
    print("\n⚡ Generating Energy Optimization Dataset...")
    
    n = 5000
    
    # Input features
    temperature = np.random.uniform(1380, 1480, n)
    hour_of_day = np.random.randint(0, 24, n)
    ambient_temp = 20 + 10 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 3, n)
    furnace_age_days = np.random.randint(0, 365, n)
    load_factor = np.random.uniform(0.6, 1.0, n)
    
    # Calculate optimal energy
    base_energy = 450
    temp_factor = 0.5 * np.abs(temperature - 1420)
    ambient_factor = 0.3 * (30 - ambient_temp)
    age_factor = 0.01 * furnace_age_days
    load_factor_effect = 50 * (1 - load_factor)
    
    optimal_energy = base_energy + temp_factor + ambient_factor + age_factor + load_factor_effect
    actual_energy = optimal_energy + np.random.normal(0, 10, n)
    
    savings_potential = np.clip(actual_energy - optimal_energy, 0, 50)
    savings_pct = (savings_potential / actual_energy) * 100
    
    df = pd.DataFrame({
        'temperature': np.round(temperature, 2),
        'hour_of_day': hour_of_day,
        'ambient_temp': np.round(ambient_temp, 1),
        'furnace_age_days': furnace_age_days,
        'load_factor': np.round(load_factor, 3),
        'actual_energy_kwh': np.round(actual_energy, 2),
        'optimal_energy_kwh': np.round(optimal_energy, 2),
        'savings_potential_kwh': np.round(savings_potential, 2),
        'savings_pct': np.round(savings_pct, 2),
        'co2_reduction_kg': np.round(savings_potential * 0.5, 2)
    })
    
    filepath = f"{CONFIG['output_dir']}/energy_optimization.csv"
    df.to_csv(filepath, index=False)
    print(f"   ✅ Saved: {filepath} ({len(df)} records)")
    return df


# ============================================================================
# 5. MAINTENANCE SCHEDULE DATASET
# ============================================================================

def generate_maintenance_data():
    """Generate maintenance and equipment health data"""
    print("\n🔧 Generating Maintenance Dataset...")
    
    n = 1000
    timestamps = pd.date_range(start=CONFIG['start_date'], periods=n, freq='D')
    
    equipment = ['Furnace_1', 'Furnace_2', 'Furnace_3', 'Sensor_Array', 
                 'Cooling_System', 'Power_Unit', 'Control_Panel', 'Safety_System']
    
    maintenance_types = ['PREVENTIVE', 'CORRECTIVE', 'PREDICTIVE', 'EMERGENCY']
    
    df = pd.DataFrame({
        'date': timestamps,
        'equipment': np.random.choice(equipment, n),
        'maintenance_type': np.random.choice(maintenance_types, n, p=[0.5, 0.25, 0.2, 0.05]),
        'duration_hours': np.round(np.random.exponential(4, n), 1),
        'cost_usd': np.round(np.random.exponential(500, n), 2),
        'downtime_hours': np.round(np.random.exponential(2, n), 1),
        'parts_replaced': np.random.randint(0, 5, n),
        'health_score_before': np.round(np.random.uniform(50, 90, n), 1),
        'health_score_after': np.round(np.random.uniform(85, 100, n), 1),
        'next_maintenance_days': np.random.randint(7, 90, n),
        'technician_id': np.random.randint(1, 10, n)
    })
    
    filepath = f"{CONFIG['output_dir']}/maintenance_history.csv"
    df.to_csv(filepath, index=False)
    print(f"   ✅ Saved: {filepath} ({len(df)} records)")
    return df


# ============================================================================
# 6. AI TRAINING CONVERSATIONS DATASET
# ============================================================================

def generate_ai_training_data():
    """Generate Q&A pairs for AI assistant training"""
    print("\n🤖 Generating AI Training Dataset...")
    
    qa_pairs = [
        # Temperature queries
        {"query": "what is the current temperature", "intent": "temperature", "entities": ["current"]},
        {"query": "how hot is the furnace", "intent": "temperature", "entities": ["current"]},
        {"query": "temperature status", "intent": "temperature", "entities": ["status"]},
        {"query": "is the temp too high", "intent": "temperature", "entities": ["high", "check"]},
        {"query": "temp reading", "intent": "temperature", "entities": ["current"]},
        
        # Prediction queries
        {"query": "what will temperature be in 30 minutes", "intent": "prediction", "entities": ["30min"]},
        {"query": "next 30 min forecast", "intent": "prediction", "entities": ["30min"]},
        {"query": "predict temperature", "intent": "prediction", "entities": ["general"]},
        {"query": "future temperature", "intent": "prediction", "entities": ["general"]},
        {"query": "next hour prediction", "intent": "prediction", "entities": ["1hour"]},
        {"query": "5 minute forecast", "intent": "prediction", "entities": ["5min"]},
        {"query": "what will happen next", "intent": "prediction", "entities": ["general"]},
        
        # Pouring queries
        {"query": "can I pour now", "intent": "pouring", "entities": ["readiness"]},
        {"query": "pouring readiness", "intent": "pouring", "entities": ["readiness"]},
        {"query": "is it safe to pour", "intent": "pouring", "entities": ["safety", "readiness"]},
        {"query": "ready to cast", "intent": "pouring", "entities": ["readiness"]},
        {"query": "pour status", "intent": "pouring", "entities": ["status"]},
        {"query": "when can I pour", "intent": "pouring", "entities": ["timing"]},
        
        # Energy queries
        {"query": "energy consumption", "intent": "energy", "entities": ["current"]},
        {"query": "power usage", "intent": "energy", "entities": ["current"]},
        {"query": "energy savings", "intent": "energy", "entities": ["savings"]},
        {"query": "how much energy are we using", "intent": "energy", "entities": ["current"]},
        {"query": "electricity cost", "intent": "energy", "entities": ["cost"]},
        {"query": "optimize energy", "intent": "energy", "entities": ["optimization"]},
        
        # Anomaly queries
        {"query": "any anomalies", "intent": "anomaly", "entities": ["check"]},
        {"query": "risk level", "intent": "anomaly", "entities": ["risk"]},
        {"query": "is there a problem", "intent": "anomaly", "entities": ["check"]},
        {"query": "system alerts", "intent": "anomaly", "entities": ["alerts"]},
        {"query": "warning status", "intent": "anomaly", "entities": ["status"]},
        {"query": "danger check", "intent": "anomaly", "entities": ["safety"]},
        
        # Status queries
        {"query": "system status", "intent": "status", "entities": ["general"]},
        {"query": "give me a report", "intent": "status", "entities": ["report"]},
        {"query": "overview", "intent": "status", "entities": ["general"]},
        {"query": "dashboard summary", "intent": "status", "entities": ["summary"]},
        {"query": "how is everything", "intent": "status", "entities": ["general"]},
        
        # Safety queries
        {"query": "safety status", "intent": "safety", "entities": ["status"]},
        {"query": "is it safe", "intent": "safety", "entities": ["check"]},
        {"query": "hazard check", "intent": "safety", "entities": ["hazard"]},
        {"query": "emergency status", "intent": "safety", "entities": ["emergency"]},
        
        # Maintenance queries
        {"query": "maintenance schedule", "intent": "maintenance", "entities": ["schedule"]},
        {"query": "when is next maintenance", "intent": "maintenance", "entities": ["next"]},
        {"query": "equipment health", "intent": "maintenance", "entities": ["health"]},
        {"query": "sensor status", "intent": "maintenance", "entities": ["sensors"]},
        
        # Optimization queries
        {"query": "how to optimize", "intent": "optimization", "entities": ["general"]},
        {"query": "improve efficiency", "intent": "optimization", "entities": ["efficiency"]},
        {"query": "reduce costs", "intent": "optimization", "entities": ["cost"]},
        {"query": "best settings", "intent": "optimization", "entities": ["settings"]},
        
        # Greetings
        {"query": "hello", "intent": "greeting", "entities": []},
        {"query": "hi", "intent": "greeting", "entities": []},
        {"query": "hey", "intent": "greeting", "entities": []},
        {"query": "good morning", "intent": "greeting", "entities": ["morning"]},
        
        # Help queries
        {"query": "help", "intent": "help", "entities": []},
        {"query": "what can you do", "intent": "help", "entities": ["capabilities"]},
        {"query": "commands", "intent": "help", "entities": ["commands"]},
        {"query": "features", "intent": "help", "entities": ["features"]},
    ]
    
    df = pd.DataFrame(qa_pairs)
    filepath = f"{CONFIG['output_dir']}/ai_training_intents.json"
    
    with open(filepath, 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    
    print(f"   ✅ Saved: {filepath} ({len(qa_pairs)} intents)")
    return qa_pairs


# ============================================================================
# 7. SHIFT PERFORMANCE DATASET
# ============================================================================

def generate_shift_performance():
    """Generate shift-wise performance metrics"""
    print("\n📈 Generating Shift Performance Dataset...")
    
    n_days = 365
    shifts = ['day', 'evening', 'night']
    
    records = []
    start_date = datetime.strptime(CONFIG['start_date'], '%Y-%m-%d')
    
    for day in range(n_days):
        date = start_date + timedelta(days=day)
        for shift in shifts:
            records.append({
                'date': date.strftime('%Y-%m-%d'),
                'shift': shift,
                'avg_temperature': round(np.random.normal(1420, 10), 2),
                'temp_variance': round(np.random.uniform(5, 25), 2),
                'energy_consumed_kwh': round(np.random.normal(3600, 200), 2),
                'pours_completed': np.random.randint(3, 8),
                'successful_pours': np.random.randint(2, 8),
                'anomalies_detected': np.random.randint(0, 5),
                'alerts_triggered': np.random.randint(0, 10),
                'downtime_minutes': np.random.randint(0, 60),
                'efficiency_score': round(np.random.uniform(75, 98), 1),
                'safety_incidents': np.random.choice([0, 0, 0, 0, 1], p=[0.95, 0.01, 0.01, 0.01, 0.02])
            })
    
    df = pd.DataFrame(records)
    filepath = f"{CONFIG['output_dir']}/shift_performance.csv"
    df.to_csv(filepath, index=False)
    print(f"   ✅ Saved: {filepath} ({len(df)} records)")
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n🚀 Starting Dataset Generation...\n")
    
    # Generate all datasets
    sensor_df = generate_sensor_fusion_data()
    pouring_df = generate_pouring_events()
    alert_df = generate_alert_history()
    energy_df = generate_energy_optimization()
    maintenance_df = generate_maintenance_data()
    ai_data = generate_ai_training_data()
    shift_df = generate_shift_performance()
    
    # Summary
    print("\n" + "═"*60)
    print("✅ DATASET GENERATION COMPLETE!")
    print("═"*60)
    print(f"\n📁 Output Directory: {CONFIG['output_dir']}")
    print(f"\n📊 Generated Datasets:")
    print(f"   • Sensor Fusion: {len(sensor_df):,} records")
    print(f"   • Pouring Events: {len(pouring_df):,} records")
    print(f"   • Alert History: {len(alert_df):,} records")
    print(f"   • Energy Optimization: {len(energy_df):,} records")
    print(f"   • Maintenance History: {len(maintenance_df):,} records")
    print(f"   • AI Training Intents: {len(ai_data)} intents")
    print(f"   • Shift Performance: {len(shift_df):,} records")
    print(f"\n🔥 Total Records: {len(sensor_df) + len(pouring_df) + len(alert_df) + len(energy_df) + len(maintenance_df) + len(shift_df):,}")
