// Theme Selection Setup
const themeToggleBtn = document.getElementById('theme-toggle');
const htmlEl = document.documentElement;

// Load initial theme
const savedTheme = localStorage.getItem('theme') || 'dark';
htmlEl.setAttribute('data-theme', savedTheme);
updateThemeIcon(savedTheme);

themeToggleBtn.addEventListener('click', () => {
    const currentTheme = htmlEl.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    htmlEl.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
});

function updateThemeIcon(theme) {
    const icon = themeToggleBtn.querySelector('i');
    if (theme === 'dark') {
        icon.className = 'fa-solid fa-sun';
    } else {
        icon.className = 'fa-solid fa-moon';
    }
}

// Variables to track state changes
let lastLivenessStatus = "PENDING";
let eventSource = null;

// Server-Sent Events (SSE) Stream Setup
function startStatusStream() {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource('/status_feed');

    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };

    eventSource.onerror = function(err) {
        console.error("SSE Connection Error:", err);
        setTimeout(startStatusStream, 2000); // Re-try connection
    };
}

// Update UI elements with live metrics
function updateDashboard(data) {
    // 1. Face Detected mini-metric
    const faceCard = document.getElementById('card-face');
    const valFace = document.getElementById('val-face');
    if (data.face_detected) {
        valFace.innerText = "Yes";
        faceCard.className = "mini-metric-item success-state";
    } else {
        valFace.innerText = "No";
        faceCard.className = "mini-metric-item";
    }

    // 2. Blinks registered mini-metric
    const blinksVal = document.getElementById('val-blinks');
    blinksVal.innerText = data.blink_count;
    const blinkCard = document.getElementById('card-blink');
    if (data.blink_count >= 1) {
        blinkCard.className = "mini-metric-item success-state";
    } else {
        blinkCard.className = "mini-metric-item";
    }

    // 3. Smile check mini-metric
    const smileVal = document.getElementById('val-smile');
    smileVal.innerText = data.smile_detected ? "Yes" : "No";
    const smileCard = document.getElementById('card-smile');
    if (data.smile_detected) {
        smileCard.className = "mini-metric-item success-state";
    } else {
        smileCard.className = "mini-metric-item";
    }

    // 4. Spoof score & Radial progress gauge
    const spoofVal = document.getElementById('val-spoof');
    spoofVal.innerText = `${data.spoof_score}%`;
    
    const spoofCard = document.getElementById('card-spoof');
    if (data.spoof_score >= 50) {
        spoofCard.className = "mini-metric-item danger-state";
    } else {
        spoofCard.className = "mini-metric-item";
    }

    // SVG Circular Progress offset logic
    const spoofCircle = document.getElementById('spoof-progress-circle');
    const radialScoreLabel = document.getElementById('val-spoof-gauge');
    if (spoofCircle) {
        const radius = spoofCircle.r.baseVal.value;
        const circumference = 2 * Math.PI * radius; // 301.59
        const offset = circumference - (data.spoof_score / 100) * circumference;
        spoofCircle.style.strokeDashoffset = offset;
        radialScoreLabel.innerText = `${data.spoof_score}%`;
        
        // Dynamic gauge color based on threat level
        if (data.spoof_score >= 50) {
            spoofCircle.setAttribute('stroke', 'var(--danger)');
            radialScoreLabel.style.color = 'var(--danger)';
        } else if (data.spoof_score >= 20) {
            spoofCircle.setAttribute('stroke', 'var(--warning)');
            radialScoreLabel.style.color = 'var(--warning)';
        } else {
            spoofCircle.setAttribute('stroke', 'var(--success)');
            radialScoreLabel.style.color = 'var(--success)';
        }
    }

    // 5. Checklist Challenges
    updateChecklistItem('check-blink', data.blink_count >= 1);
    updateChecklistItem('check-left', data.left_turn);
    updateChecklistItem('check-right', data.right_turn);

    // 6. Liveness Header Banner
    const banner = document.getElementById('liveness-banner');
    const bannerIcon = document.getElementById('banner-icon');
    const bannerTitle = document.getElementById('liveness-title');
    const bannerDesc = document.getElementById('liveness-desc');
    const timerVal = document.getElementById('timer-val');

    timerVal.innerText = `${data.time_left}s`;

    if (data.liveness_status === "LIVE PERSON") {
        banner.className = "status-banner-badge live";
        bannerIcon.className = "fa-solid fa-circle-check";
        bannerTitle.innerText = "LIVE PERSON VERIFIED";
        bannerDesc.innerText = "Secure authentication passed. Access granted.";
        timerVal.style.display = "none";
    } else if (data.liveness_status === "SPOOF DETECTED") {
        banner.className = "status-banner-badge spoof";
        bannerIcon.className = "fa-solid fa-triangle-exclamation";
        bannerTitle.innerText = "SPOOF DETECTED";
        bannerDesc.innerText = "Verification halted. Security intercept active.";
        timerVal.style.display = "none";
    } else {
        banner.className = "status-banner-badge pending";
        bannerIcon.className = "fa-solid fa-circle-notch fa-spin";
        bannerTitle.innerText = "VERIFYING AUTHENTICITY";
        bannerDesc.innerText = "Blink eyes and rotate head left or right";
        timerVal.style.display = "block";
    }

    // 7. Identity Recognition Profile Card
    const identityName = document.getElementById('recognized-identity');
    const identityConf = document.getElementById('recognized-conf');
    const statusBadge = document.getElementById('recog-status-badge');

    identityName.innerText = data.recognized_name;

    if (data.recognized_name !== "Unknown") {
        identityConf.innerHTML = `<i class="fa-solid fa-circle-nodes"></i> ${data.recognition_confidence}% confidence`;
        statusBadge.innerText = "Verified";
        statusBadge.className = "verification-badge verified";
    } else {
        identityConf.innerHTML = `<i class="fa-solid fa-circle-nodes"></i> 0% confidence`;
        statusBadge.innerText = "Unregistered";
        statusBadge.className = "verification-badge unverified";
    }

    // 8. Auto Reload on status changes
    if (data.liveness_status !== lastLivenessStatus) {
        if (data.liveness_status === "LIVE PERSON" || data.liveness_status === "SPOOF DETECTED") {
            loadAttendanceLogs();
            loadDashboardStats();
        }
        lastLivenessStatus = data.liveness_status;
    }
}

function updateChecklistItem(elementId, isDone) {
    const item = document.getElementById(elementId);
    const checkbox = item.querySelector('.checkbox-icon');
    if (isDone) {
        item.classList.add('done');
        checkbox.className = "fa-solid fa-circle-check checkbox-icon";
    } else {
        item.classList.remove('done');
        checkbox.className = "fa-regular fa-circle checkbox-icon";
    }
}

// Reset Liveness State on backend
document.getElementById('btn-reset').addEventListener('click', () => {
    fetch('/reset', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            console.log(data.message);
            lastLivenessStatus = "PENDING";
            // Refresh logs and statistics
            loadAttendanceLogs();
            loadDashboardStats();
        })
        .catch(err => console.error("Reset error:", err));
});

// Enroll User handler
function enrollUser() {
    const nameInput = document.getElementById('enroll-name');
    const name = nameInput.value.trim();
    const statusMsg = document.getElementById('enrollment-status');

    if (!name) return;

    statusMsg.style.color = "var(--warning)";
    statusMsg.innerText = "Initializing biometric camera... Look straight.";

    fetch('/enroll', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "success") {
            statusMsg.style.color = "var(--primary)";
            statusMsg.innerText = "Enrolling profile. Blink and rotate head...";
            nameInput.value = "";
            
            // Poll for enrollment training finish (simulated 15s check)
            let attempts = 0;
            const interval = setInterval(() => {
                attempts++;
                if (attempts > 30) {
                    clearInterval(interval);
                    statusMsg.style.color = "var(--danger)";
                    statusMsg.innerText = "Enrollment session timed out.";
                }
                
                if (attempts === 15) {
                    clearInterval(interval);
                    statusMsg.style.color = "var(--success)";
                    statusMsg.innerText = `Profile successfully registered for ${name}!`;
                    loadDashboardStats();
                    setTimeout(() => { statusMsg.innerText = ""; }, 4000);
                }
            }, 1000);
        } else {
            statusMsg.style.color = "var(--danger)";
            statusMsg.innerText = `Error: ${data.message}`;
        }
    })
    .catch(err => {
        statusMsg.style.color = "var(--danger)";
        statusMsg.innerText = "Failed to communicate with enrollment API.";
        console.error("Enroll error:", err);
    });
}

// Fetch dynamic statistics counts
function loadDashboardStats() {
    fetch('/stats')
        .then(res => res.json())
        .then(data => {
            document.getElementById('stat-total').innerText = data.total_verifications;
            document.getElementById('stat-registered').innerText = data.registered_users;
            document.getElementById('stat-live').innerText = data.live_verifications;
            document.getElementById('stat-spoof').innerText = data.spoof_attempts;
        })
        .catch(err => console.error("Error loading stats:", err));
}

// Load attendance log table records
function loadAttendanceLogs() {
    fetch('/attendance')
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById('attendance-tbody');
            
            // Update Report timestamp of last log
            const reportTimestamp = document.getElementById('report-timestamp');
            if (data.length > 0) {
                reportTimestamp.innerText = `Last generated: ${data[0].timestamp}`;
            } else {
                reportTimestamp.innerText = "Last generated: Just now";
            }

            if (data.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 24px;">No verification records found</td>
                    </tr>
                `;
                return;
            }

            tbody.innerHTML = "";
            data.forEach(log => {
                const tr = document.createElement('tr');
                const badgeClass = log.liveness === "LIVE PERSON" ? "tbl-badge live" : "tbl-badge spoof";
                
                // Rotations text clean-up
                const rotationInfo = `Left: ${log.left_turn} | Right: ${log.right_turn}`;

                tr.innerHTML = `
                    <td>${log.timestamp}</td>
                    <td><strong>${log.name}</strong></td>
                    <td>${log.blinks}</td>
                    <td>${rotationInfo}</td>
                    <td><span class="${badgeClass}">${log.liveness}</span></td>
                    <td>
                        <button class="btn-tbl-action btn-edit" onclick="editAttendanceName('${log.timestamp}', '${log.name}')" title="Edit Name">
                            <i class="fa-solid fa-pen"></i>
                        </button>
                        <button class="btn-tbl-action btn-delete" onclick="deleteAttendanceRecord('${log.timestamp}', '${log.name}')" title="Delete Entry">
                            <i class="fa-solid fa-trash"></i>
                        </button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(err => console.error("Error loading logs:", err));
}

// Edit attendance name handler
function editAttendanceName(timestamp, oldName) {
    const newName = prompt(`Edit identity name for record at ${timestamp}:`, oldName);
    if (newName === null) return;
    const trimmed = newName.trim();
    if (!trimmed) {
        alert("Identity name cannot be empty.");
        return;
    }
    
    fetch('/edit_attendance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timestamp: timestamp, old_name: oldName, new_name: trimmed })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "success") {
            loadAttendanceLogs();
            loadDashboardStats();
        } else {
            alert(`Error: ${data.message}`);
        }
    })
    .catch(err => console.error("Edit attendance error:", err));
}

// Delete attendance record handler
function deleteAttendanceRecord(timestamp, name) {
    if (!confirm(`Are you sure you want to permanently delete the audit log for "${name}" at ${timestamp}?`)) {
        return;
    }
    
    fetch('/delete_attendance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timestamp: timestamp, name: name })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "success") {
            loadAttendanceLogs();
            loadDashboardStats();
        } else {
            alert(`Error: ${data.message}`);
        }
    })
    .catch(err => console.error("Delete attendance error:", err));
}

// Client-Side Search logs filter
document.getElementById('table-search').addEventListener('input', function(e) {
    const query = e.target.value.toLowerCase().trim();
    const rows = document.querySelectorAll('#attendance-tbody tr');

    rows.forEach(row => {
        // Skip default empty placeholder row
        if (row.cells.length === 1 && row.cells[0].colSpan === 6) return;
        
        const timestampText = row.cells[0].innerText.toLowerCase();
        const nameText = row.cells[1].innerText.toLowerCase();
        const livenessText = row.cells[4].innerText.toLowerCase();

        if (timestampText.includes(query) || nameText.includes(query) || livenessText.includes(query)) {
            row.style.display = "";
        } else {
            row.style.display = "none";
        }
    });
});

// Initialization
window.addEventListener('DOMContentLoaded', () => {
    startStatusStream();
    loadDashboardStats();
    loadAttendanceLogs();
});
