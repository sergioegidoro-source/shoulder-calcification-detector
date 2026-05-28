document.addEventListener("DOMContentLoaded", () => {
    
    // 1. Initialize Stats Loading
    loadLandingStats();

    // 2. Attach Event Listeners to Cards
    const btnStandard = document.getElementById('btn-standard');
    const btnAdvanced = document.getElementById('btn-advanced');

    if (btnStandard) {
        btnStandard.addEventListener('click', () => {
            selectModel('COMBINED_512');
        });
    }

    if (btnAdvanced) {
        btnAdvanced.addEventListener('click', () => {
            selectModel('SEG_300');
        });
    }
});

// Helper function for selection and redirection
function selectModel(modelName) {
    // Save preference
    localStorage.setItem("selectedModel", modelName);
    // Redirect to the App page
    window.location.href = "model.html"; 
}

// Stats Loading Logic
async function loadLandingStats() {
    try {
        const response = await fetch('/stats');
        
        // If the server is down or returns 404/500, stop here
        if (!response.ok) return;

        const data = await response.json();

        // Helper function to safely update text content
        const setTxt = (id, val) => {
            const el = document.getElementById(id);
            if(el) el.textContent = val;
        };

        // Update UI
        setTxt('val-cnn', data.models?.cnn || 0);
        setTxt('val-seg', data.models?.seg || 0);
        setTxt('val-hybrid', data.models?.hybrid || 0);

        setTxt('val-left', data.laterality?.Left || 0);
        setTxt('val-right', data.laterality?.Right || 0);
        setTxt('val-unknown', data.laterality?.Unknown || 0);

        setTxt('val-pos', data.diagnosis?.Positive || 0);
        setTxt('val-neg', data.diagnosis?.Negative || 0);

    } catch (error) {
        console.error("Error loading stats:", error);
    }
}