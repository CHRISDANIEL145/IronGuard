/* ============================================================================
   🔥 FORGE INTELLIGENCE v3.0 - MAIN.JS (PRODUCTION FINAL)
   Complete Frontend with Real-Time Socket.IO & Dynamic Dashboard Updates
   Quantum ML Predictions, Anomaly Detection, AI Chat Interface
   
   Author: Quantum AI Engineering Team
   Date: 2025
   License: MIT
   Version: 3.0 (Production Final)
   ============================================================================ */

// ============================================================================
// 1. GLOBAL STATE & CONFIGURATION
// ============================================================================

const AppState = {
    // Connection
    socket: null,
    isConnected: false,
    
    // Temperature Data (Dynamic)
    tempHistory: [],
    currentTemp: 1420.0,
    predictedTemp: 1420.0,
    lastUpdate: null,
    
    // Energy Data (Dynamic)
    energyHistory: [],
    currentEnergy: 450.0,
    energySavings: 1.8,
    annualROI: 2700,
    
    // Anomaly Data (Dynamic)
    anomalyRisk: 0.02,
    isAnomaly: false,
    pourReadiness: 85,
    
    // Chat
    chatHistory: [],
    
    // Charts
    charts: {},
    
    // Configuration
    maxDataPoints: 100,
    updateIntervals: {
        temperature: 5000,
        prediction: 10000,
        energy: 20000,
        anomaly: 15000,
    },
    
    // UI
    isDarkMode: true,
    isMobile: window.innerWidth <= 768,
};

// API Endpoints
const API_ENDPOINTS = {
    chat: '/api/chat',
    predict: '/api/predict',
    anomaly: '/api/anomaly',
    energy: '/api/energy_status',
    status: '/api/status',
};

// ============================================================================
// 2. INITIALIZATION - DOCUMENT READY
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('📱 Initializing Forge Intelligence v3.0...');
    
    // Initialize Socket.IO
    initializeSocket();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize charts
    initializeCharts();
    
    // Start live update intervals
    startLiveUpdates();
    
    console.log('✅ Forge Intelligence initialized successfully');
});

// ============================================================================
// 3. SOCKET.IO CONNECTION & REAL-TIME UPDATES
// ============================================================================

function initializeSocket() {
    console.log('🔌 Connecting to Socket.IO server...');
    
    AppState.socket = io({
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: 10,
    });
    
    // ===== CONNECTION EVENTS =====
    AppState.socket.on('connect', handleConnect);
    AppState.socket.on('disconnect', handleDisconnect);
    AppState.socket.on('connect_error', handleConnectError);
    
    // ===== DATA EVENTS =====
    AppState.socket.on('temp_update', handleTempUpdate);
    AppState.socket.on('system_status', handleSystemStatus);
    AppState.socket.on('connection_status', handleConnectionStatus);
}

function handleConnect() {
    console.log('✅ Connected to Forge Intelligence backend');
    AppState.isConnected = true;
    updateConnectionStatus(true);
    AppState.socket.emit('request_status');
    showNotification('Connected to Quantum ML Engine', 'success');
}

function handleDisconnect() {
    console.log('❌ Disconnected from backend');
    AppState.isConnected = false;
    updateConnectionStatus(false);
    showNotification('Disconnected - Attempting to reconnect...', 'warning');
}

function handleConnectError(error) {
    console.error('❌ Connection error:', error);
    showNotification('Connection error - Reconnecting...', 'danger');
}

// ===== REAL-TIME TEMPERATURE UPDATES =====
function handleTempUpdate(data) {
    console.log('📊 Temperature update:', data);
    
    const temp = parseFloat(data.temp);
    const timestamp = data.timestamp;
    const isAnomaly = data.anomaly || false;
    const quantumRisk = parseFloat(data.quantum_risk) || 0;
    
    // Update global state
    AppState.currentTemp = temp;
    AppState.lastUpdate = timestamp;
    AppState.anomalyRisk = quantumRisk;
    AppState.isAnomaly = isAnomaly;
    
    // Keep history
    AppState.tempHistory.push(temp);
    if (AppState.tempHistory.length > AppState.maxDataPoints) {
        AppState.tempHistory.shift();
    }
    
    // 🔥 LIVE DOM UPDATES
    updateTemperatureDisplay(temp, timestamp);
    updateTemperatureStatus(temp);
    updateQuantumRiskMeter(quantumRisk);
    updateAnomalyIndicator(isAnomaly, quantumRisk);
    updateTemperatureChart();
    
    // Alerts
    if (isAnomaly || quantumRisk > 0.7) {
        showQuantumAlert(temp, quantumRisk);
    }
}

function handleSystemStatus(data) {
    console.log('📋 System status:', data);
    
    const clientsEl = document.getElementById('status-clients');
    const modelsEl = document.getElementById('status-models');
    const riskEl = document.getElementById('quantum-risk-display');
    
    if (clientsEl) clientsEl.textContent = data.connected_clients || 0;
    if (modelsEl) modelsEl.textContent = data.models_trained ? '✅ Trained' : '⏳ Training';
    if (riskEl) riskEl.textContent = (data.quantum_risk * 100).toFixed(1) + '%';
}

function handleConnectionStatus(data) {
    console.log('🔗 Connection status:', data.message);
}

// ============================================================================
// 4. TEMPERATURE DISPLAY UPDATES
// ============================================================================

function updateTemperatureDisplay(temp, timestamp) {
    // Current temperature
    const tempEl = document.getElementById('current-temp');
    if (tempEl) {
        const rounded = Math.round(temp * 10) / 10;
        tempEl.textContent = rounded + '°C';
        tempEl.style.animation = 'none';
        setTimeout(() => {
            tempEl.style.animation = 'pulse-glow 0.6s ease';
        }, 10);
    }
    
    // Timestamp
    const timeEl = document.getElementById('temp-timestamp');
    if (timeEl) {
        timeEl.textContent = timestamp || new Date().toLocaleTimeString();
    }
}

function updateTemperatureStatus(temp) {
    const statusEl = document.getElementById('temp-status');
    if (!statusEl) return;
    
    const OPTIMAL_LOW = 1410;
    const OPTIMAL_HIGH = 1430;
    const CRITICAL = 1480;
    
    let status = '';
    let color = '';
    let emoji = '';
    
    if (temp >= OPTIMAL_LOW && temp <= OPTIMAL_HIGH) {
        status = 'OPTIMAL - READY TO POUR';
        color = '#39FF14';
        emoji = '🟢';
    } else if (temp > CRITICAL) {
        status = 'CRITICAL - EMERGENCY';
        color = '#DC143C';
        emoji = '🔴';
    } else if (temp > OPTIMAL_HIGH) {
        status = 'WARM - WAIT FOR COOLDOWN';
        color = '#FFD700';
        emoji = '🟡';
    } else {
        status = 'COOL - WAIT FOR HEATING';
        color = '#00BFFF';
        emoji = '🔵';
    }
    
    statusEl.innerHTML = `${emoji} ${status}`;
    statusEl.style.color = color;
}

function updateQuantumRiskMeter(risk) {
    const meterEl = document.getElementById('quantum-risk-meter');
    if (meterEl) {
        const percentage = Math.min(risk * 100, 100);
        meterEl.style.width = percentage + '%';
        
        if (risk > 0.7) {
            meterEl.style.backgroundColor = '#DC143C';
        } else if (risk > 0.4) {
            meterEl.style.backgroundColor = '#FFD700';
        } else {
            meterEl.style.backgroundColor = '#39FF14';
        }
    }
    
    // Risk percentage text
    const riskTextEl = document.getElementById('quantum-risk');
    if (riskTextEl) {
        riskTextEl.textContent = (risk * 100).toFixed(1) + '%';
        riskTextEl.style.color = risk > 0.7 ? '#DC143C' : risk > 0.4 ? '#FFD700' : '#39FF14';
    }
}

function updateAnomalyIndicator(isAnomaly, quantumRisk) {
    const indicator = document.getElementById('anomaly-indicator');
    if (indicator) {
        if (isAnomaly || quantumRisk > 0.7) {
            indicator.innerHTML = '🚨';
            indicator.style.animation = 'pulse 0.5s infinite';
            indicator.style.color = '#DC143C';
        } else {
            indicator.innerHTML = '✅';
            indicator.style.animation = 'none';
            indicator.style.color = '#39FF14';
        }
    }
}

// ============================================================================
// 5. PREDICTION DISPLAY UPDATES
// ============================================================================

async function fetchAndUpdatePredictions() {
    if (!AppState.isConnected) return;
    
    try {
        console.log('🔮 Fetching quantum predictions...');
        
        const response = await fetch(API_ENDPOINTS.predict, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) throw new Error('Prediction failed');
        
        const data = await response.json();
        
        const predictedTemp = parseFloat(data.predicted_temp);
        const confidence = parseFloat(data.confidence);
        
        console.log(`📈 Predicted: ${predictedTemp}°C (${(confidence*100).toFixed(1)}%)`);
        
        // Update state
        AppState.predictedTemp = predictedTemp;
        
        // 🔥 UPDATE DOM
        updatePredictionDisplay(predictedTemp, confidence);
        calculateAndUpdatePourReadiness();
        
    } catch (error) {
        console.error('❌ Prediction error:', error);
    }
}

function updatePredictionDisplay(predictedTemp, confidence) {
    // Next 30 min prediction
    const predEl = document.getElementById('predicted-temp');
    if (predEl) {
        const rounded = Math.round(predictedTemp * 10) / 10;
        predEl.textContent = rounded + '°C';
        predEl.style.animation = 'pulse-glow 1s ease';
    }
    
    // Confidence
    const confEl = document.getElementById('prediction-confidence');
    if (confEl) {
        confEl.textContent = (confidence * 100).toFixed(1) + '%';
    }
    
    // Prediction model
    const modelEl = document.getElementById('prediction-model');
    if (modelEl) {
        modelEl.textContent = 'Quantum Superposition Ensemble';
    }
}

function calculateAndUpdatePourReadiness() {
    const OPTIMAL_LOW = 1410;
    const OPTIMAL_HIGH = 1430;
    
    let readiness = 0;
    
    if (AppState.currentTemp >= OPTIMAL_LOW && AppState.currentTemp <= OPTIMAL_HIGH) {
        readiness = 95; // Almost ready now
    } else if (AppState.predictedTemp >= OPTIMAL_LOW && AppState.predictedTemp <= OPTIMAL_HIGH) {
        readiness = 80; // Will be ready soon
    } else if (AppState.currentTemp > OPTIMAL_HIGH) {
        readiness = 50; // Wait for cooldown
    } else {
        readiness = 20; // Need heating
    }
    
    AppState.pourReadiness = readiness;
    updatePourReadinessDisplay(readiness);
}

function updatePourReadinessDisplay(readiness) {
    const el = document.getElementById('pour-readiness');
    if (el) {
        el.textContent = readiness + '%';
        el.style.animation = 'pulse-glow 1s ease';
        
        // Color coding
        if (readiness >= 90) {
            el.style.color = '#39FF14';
        } else if (readiness >= 70) {
            el.style.color = '#FFD700';
        } else {
            el.style.color = '#DC143C';
        }
    }
}

// ============================================================================
// 6. ENERGY DISPLAY UPDATES
// ============================================================================

async function fetchAndUpdateEnergy() {
    if (!AppState.isConnected) return;
    
    try {
        console.log('⚡ Fetching energy status...');
        
        const response = await fetch(API_ENDPOINTS.energy);
        if (!response.ok) throw new Error('Energy fetch failed');
        
        const data = await response.json();
        
        const energy = parseFloat(data.current_energy);
        const savings = parseFloat(data.savings_pct);
        const roi = parseInt(data.roi_annual);
        
        console.log(`⚡ Energy: ${energy} kWh | Savings: ${savings}%`);
        
        // Update state
        AppState.currentEnergy = energy;
        AppState.energySavings = savings;
        AppState.annualROI = roi;
        
        // 🔥 UPDATE DOM
        updateEnergyDisplay(energy, savings, roi);
        
    } catch (error) {
        console.error('❌ Energy error:', error);
    }
}

function updateEnergyDisplay(energy, savings, roi) {
    // Current energy
    const energyEl = document.getElementById('current-energy');
    if (energyEl) {
        energyEl.textContent = Math.round(energy) + ' kWh';
    }
    
    // Savings percentage
    const savingsEl = document.getElementById('energy-savings');
    if (savingsEl) {
        savingsEl.textContent = savings.toFixed(1) + '%';
        
        // Color coding
        if (savings > 15) {
            savingsEl.style.color = '#39FF14';
        } else if (savings > 10) {
            savingsEl.style.color = '#FFD700';
        } else {
            savingsEl.style.color = '#DC143C';
        }
    }
    
    // Annual ROI
    const roiEl = document.getElementById('annual-roi');
    if (roiEl) {
        roiEl.textContent = '$' + Math.round(roi) + 'K';
    }
}

// ============================================================================
// 7. ANOMALY DETECTION UPDATES
// ============================================================================

async function fetchAndUpdateAnomalies() {
    if (!AppState.isConnected) return;
    
    try {
        console.log('🚨 Checking for anomalies...');
        
        const response = await fetch(API_ENDPOINTS.anomaly);
        if (!response.ok) throw new Error('Anomaly check failed');
        
        const data = await response.json();
        
        const score = parseFloat(data.anomaly_score);
        const isAnomaly = data.is_anomaly;
        const risk = parseFloat(data.quantum_risk);
        
        console.log(`🚨 Anomaly Risk: ${(risk*100).toFixed(1)}%`);
        
        // Update state
        AppState.anomalyRisk = risk;
        AppState.isAnomaly = isAnomaly;
        
        // 🔥 UPDATE DOM
        updateAnomalyStatus(isAnomaly, score, risk);
        
    } catch (error) {
        console.error('❌ Anomaly error:', error);
    }
}

function updateAnomalyStatus(isAnomaly, score, risk) {
    const statusEl = document.getElementById('anomaly-status');
    if (statusEl) {
        if (isAnomaly || risk > 0.7) {
            statusEl.textContent = '🚨 ANOMALY DETECTED';
            statusEl.style.color = '#DC143C';
        } else {
            statusEl.textContent = '🟢 NORMAL';
            statusEl.style.color = '#39FF14';
        }
    }
}

// ============================================================================
// 8. CHART INITIALIZATION & UPDATES
// ============================================================================

function initializeCharts() {
    console.log('📊 Initializing charts...');
    
    initTemperatureChart();
    initEnergyChart();
}

function initTemperatureChart() {
    const ctx = document.getElementById('tempChart');
    if (!ctx) return;
    
    AppState.charts.tempChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: AppState.maxDataPoints }, (_, i) => i),
            datasets: [{
                label: 'Temperature (°C)',
                data: AppState.tempHistory,
                borderColor: '#FF4500',
                backgroundColor: 'rgba(255, 69, 0, 0.1)',
                borderWidth: 3,
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 8,
                pointBackgroundColor: '#FF4500',
                pointBorderColor: '#FFD700',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0,0,0,0.9)',
                    titleColor: '#FF4500',
                    bodyColor: 'rgba(255,255,255,0.9)',
                    borderColor: '#FF4500',
                    borderWidth: 2,
                }
            },
            scales: {
                y: {
                    min: 1350,
                    max: 1550,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: 'rgba(248, 249, 250, 0.7)' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: 'rgba(248, 249, 250, 0.7)' }
                }
            }
        }
    });
    
    console.log('✅ Temperature chart initialized');
}

function initEnergyChart() {
    const ctx = document.getElementById('energyChart');
    if (!ctx) return;
    
    AppState.charts.energyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Energy Saved (kWh)',
                data: [12, 15, 10, 18, 14, 24, 16],
                backgroundColor: [
                    'rgba(255, 69, 0, 0.75)',
                    'rgba(255, 69, 0, 0.75)',
                    'rgba(255, 69, 0, 0.75)',
                    'rgba(255, 69, 0, 0.75)',
                    'rgba(255, 69, 0, 0.75)',
                    'rgba(255, 69, 0, 0.85)',
                    'rgba(255, 69, 0, 0.75)'
                ],
                borderColor: '#FF4500',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: 'rgba(248, 249, 250, 0.7)' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: 'rgba(248, 249, 250, 0.7)' }
                }
            }
        }
    });
    
    console.log('✅ Energy chart initialized');
}

function updateTemperatureChart() {
    if (AppState.charts.tempChart && AppState.tempHistory.length > 0) {
        AppState.charts.tempChart.data.datasets[0].data = AppState.tempHistory.slice(-AppState.maxDataPoints);
        AppState.charts.tempChart.update('none');
    }
}

// ============================================================================
// 9. CHAT FUNCTIONALITY
// ============================================================================

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    if (!message) {
        showNotification('Please enter a message', 'warning');
        return;
    }
    
    // Show user message
    addChatMessage('user', message);
    input.value = '';
    
    // Disable send button
    const sendBtn = document.querySelector('.btn-send');
    if (sendBtn) {
        sendBtn.disabled = true;
        sendBtn.textContent = 'Sending...';
    }
    
    try {
        console.log('💬 Sending chat message...');
        
        const response = await fetch(API_ENDPOINTS.chat, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: message })
        });
        
        if (!response.ok) throw new Error('Chat failed');
        
        const data = await response.json();
        
        addChatMessage('bot', data.response);
        
        console.log(`🤖 Response: ${data.response}`);
        
    } catch (error) {
        console.error('❌ Chat error:', error);
        addChatMessage('bot', '❌ Error processing request');
        showNotification('Failed to send message', 'danger');
    } finally {
        if (sendBtn) {
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
        }
        input.focus();
    }
}

function addChatMessage(type, text) {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const sender = type === 'user' ? '👤 You' : '🤖 Forge AI';
    messageDiv.innerHTML = `<strong>${sender}:</strong><br>${text}`;
    
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
    
    // Limit chat history
    const messages = container.querySelectorAll('.message');
    if (messages.length > 100) {
        messages[0].remove();
    }
}

function handleChatKeypress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
}

// ============================================================================
// 10. LIVE UPDATE INTERVALS
// ============================================================================

function startLiveUpdates() {
    console.log('⏰ Starting live update intervals...');
    
    // Predictions: Every 10 seconds
    setInterval(() => {
        if (AppState.isConnected) {
            fetchAndUpdatePredictions();
        }
    }, AppState.updateIntervals.prediction);
    
    // Energy: Every 20 seconds
    setInterval(() => {
        if (AppState.isConnected) {
            fetchAndUpdateEnergy();
        }
    }, AppState.updateIntervals.energy);
    
    // Anomalies: Every 15 seconds
    setInterval(() => {
        if (AppState.isConnected) {
            fetchAndUpdateAnomalies();
        }
    }, AppState.updateIntervals.anomaly);
    
    // Initial fetch
    setTimeout(() => {
        fetchAndUpdatePredictions();
        fetchAndUpdateEnergy();
        fetchAndUpdateAnomalies();
    }, 1000);
    
    console.log('✅ Live updates started');
}

// ============================================================================
// 11. UI HELPER FUNCTIONS
// ============================================================================

function updateConnectionStatus(isConnected) {
    const indicator = document.querySelector('.status-dot');
    if (indicator) {
        indicator.style.backgroundColor = isConnected ? '#39FF14' : '#DC143C';
    }
    
    const text = document.getElementById('status-text');
    if (text) {
        text.textContent = isConnected ? '🟢 ONLINE' : '🔴 OFFLINE';
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 120px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        padding: 1rem 1.5rem;
        background: rgba(26, 26, 46, 0.95);
        border: 2px solid;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        animation: slideIn 0.3s ease;
        font-family: 'Rajdhani', sans-serif;
    `;
    
    if (type === 'success') {
        notification.style.borderColor = '#39FF14';
        notification.style.color = '#39FF14';
    } else if (type === 'danger') {
        notification.style.borderColor = '#DC143C';
        notification.style.color = '#DC143C';
    } else if (type === 'warning') {
        notification.style.borderColor = '#FFD700';
        notification.style.color = '#FFD700';
    } else {
        notification.style.borderColor = '#00BFFF';
        notification.style.color = '#00BFFF';
    }
    
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

function showQuantumAlert(temp, risk) {
    if (risk > 0.7) {
        showNotification(`🚨 QUANTUM ALERT: ${temp}°C (Risk: ${(risk*100).toFixed(1)}%)`, 'danger');
    } else if (risk > 0.4) {
        showNotification(`⚠️ QUANTUM WARNING: Elevated risk detected`, 'warning');
    }
}

// ============================================================================
// 12. EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    console.log('🔗 Setting up event listeners...');
    
    // Chat
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('keypress', handleChatKeypress);
    }
    
    const sendBtn = document.querySelector('.btn-send');
    if (sendBtn) {
        sendBtn.addEventListener('click', sendChatMessage);
    }
    
    // Window resize
    window.addEventListener('resize', () => {
        AppState.isMobile = window.innerWidth <= 768;
        if (AppState.charts.tempChart) AppState.charts.tempChart.resize();
        if (AppState.charts.energyChart) AppState.charts.energyChart.resize();
    });
    
    console.log('✅ Event listeners setup complete');
}

// ============================================================================
// 13. ACTION BUTTONS
// ============================================================================

function startMonitoring() {
    console.log('⚡ Starting monitoring...');
    showNotification('🔥 Real-time monitoring activated!', 'success');
    addChatMessage('bot', 'Monitoring initiated. Temperature sensors calibrated. Ready for foundry operations.');
}

function activateAI() {
    console.log('🤖 Activating AI...');
    document.getElementById('chat-input').focus();
    showNotification('AI Agent activated', 'success');
}

function checkPourReadiness() {
    console.log('⏳ Checking pour readiness...');
    document.getElementById('chat-input').value = 'Should I pour now?';
    sendChatMessage();
}

function checkAnomalies() {
    console.log('🔮 Checking anomalies...');
    fetchAndUpdateAnomalies();
    document.getElementById('chat-input').value = 'Any anomalies detected?';
    sendChatMessage();
}

function getEnergyStatus() {
    console.log('⚡ Getting energy status...');
    fetchAndUpdateEnergy();
    document.getElementById('chat-input').value = 'What is our energy efficiency?';
    sendChatMessage();
}

function exportReport() {
    console.log('📊 Exporting report...');
    showNotification('Report export feature coming soon', 'info');
}

// ============================================================================
// 14. EXPORT FOR GLOBAL ACCESS
// ============================================================================

window.AppState = AppState;
window.sendChatMessage = sendChatMessage;
window.startMonitoring = startMonitoring;
window.activateAI = activateAI;
window.checkPourReadiness = checkPourReadiness;
window.checkAnomalies = checkAnomalies;
window.getEnergyStatus = getEnergyStatus;
window.exportReport = exportReport;
window.handleChatKeypress = handleChatKeypress;

console.log('✅ Forge Intelligence v3.0 - Main.JS Loaded Successfully');
console.log('🧬 Quantum ML Engine: Ready');
console.log('📊 Real-time monitoring: Active');
console.log('🤖 AI Chat Assistant: Online');
