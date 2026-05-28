/**
 * pdf.js - Versión Final Unificada
 * Incluye: RX Hombro, Localización manguito + Confianza YOLO, 
 * Probabilidades en todos los modelos y Caja Legal Adaptable.
 */
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

async function generatePDF() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF('p', 'mm', 'a4');
    const btn = document.querySelector('.btn-generate-report');
    
    const originalContent = btn.innerHTML;
    btn.innerHTML = '<i class="fa-solid fa-cog fa-spin"></i> GENERATING...';
    btn.disabled = true;

    try {
        // --- 1. HEADER ---
        doc.setFillColor(30, 41, 59); 
        doc.rect(0, 0, 210, 25, 'F');
        doc.setFont("helvetica", "bold");
        doc.setFontSize(16);
        doc.setTextColor(255, 255, 255);
        doc.text("AI RADIOLOGY REPORT", 105, 12, { align: "center" });
        doc.setFontSize(8);
        doc.text(`ID: ${generateUUID()} | ${new Date().toLocaleString()}`, 105, 20, { align: "center" });

        // --- 2. LEGAL NOTICE (Caja Adaptable) ---
        doc.setFontSize(8);
        const legalText = "This AI-generated report is a decision-support tool, NOT a definitive diagnosis. Findings must be verified by a qualified healthcare professional. Final diagnostic responsibility remains with the attending physician.";
        const maxWidth = 180; 
        const textLines = doc.splitTextToSize(legalText, maxWidth);
        const boxHeight = (textLines.length * 4) + 8;

        doc.setFillColor(254, 243, 199);
        doc.setDrawColor(245, 158, 11);
        doc.rect(10, 30, 190, boxHeight, 'F');
        doc.rect(10, 30, 190, boxHeight, 'S');
        doc.setTextColor(180, 83, 9);
        doc.setFont("helvetica", "bold");
        doc.text("MEDICAL DISCLAIMER:", 15, 36);
        doc.setFont("helvetica", "normal");
        doc.text(textLines, 15, 41);

        let currentY = 30 + boxHeight + 8;

        // --- 3. CASE INFORMATION (Tipo de Prueba) ---
        doc.setTextColor(0, 0, 0);
        doc.setFont("helvetica", "bold");
        doc.setFontSize(11);
        doc.text("1. CASE INFORMATION", 10, currentY);
        currentY += 6;

        doc.setFontSize(9);
        doc.setFont("helvetica", "normal");
        const modelName = document.getElementById('selectedModelText')?.innerText || "Standard Pipeline";
        const laterality = document.getElementById('res-laterality')?.innerText || "--";
        
        doc.text(`• Study Type: Shoulder X-Ray (RX Hombro)`, 15, currentY);
        currentY += 5;
        doc.text(`• Clinical Model: ${modelName} | Laterality: ${laterality}`, 15, currentY);
        currentY += 8;

        // --- 4. VISUAL EVIDENCE (IMAGEN) ---
        const comparisonGrid = document.querySelector('.comparison-container');
        if (comparisonGrid) {
            const canvas = await html2canvas(comparisonGrid, {
                backgroundColor: "#0f172a",
                scale: 1.5,
                useCORS: true
            });
            const imgData = canvas.toDataURL('image/jpeg', 0.8);
            const imgWidth = 190;
            let imgHeight = (canvas.height * imgWidth) / canvas.width;
            const maxHeight = 125; // Ajuste fino para asegurar una sola página
            if (imgHeight > maxHeight) imgHeight = maxHeight;

            doc.setFont("helvetica", "bold");
            doc.text("2. VISUAL AI EVIDENCE", 10, currentY);
            currentY += 5;
            doc.addImage(imgData, 'JPEG', 10, currentY, imgWidth, imgHeight);
            currentY += imgHeight + 10; 
        }

        // --- 5. DETAILED FINDINGS (Incluyendo Probabilidades) ---
        doc.setFont("helvetica", "bold");
        doc.setFontSize(11);
        doc.text("3. DETAILED FINDINGS", 10, currentY);
        currentY += 6;

        doc.setFontSize(9);
        doc.setFont("helvetica", "normal");

        const results = [
            { id: 'res-cnn-diag', label: 'Visual Model (CNN)', probId: 'res-cnn-prob' },
            { id: 'res-hyb-diag', label: 'Pattern Model (Hybrid)', probId: 'res-hyb-prob' },
            { id: 'res-seg-diag', label: 'Segmentation (300px)', probId: 'res-seg-prob', parent: 'grid-300' }
        ];

        results.forEach(res => {
            const el = document.getElementById(res.id);
            const isVisible = res.parent ? document.getElementById(res.parent)?.style.display !== 'none' : true;
            if (el && el.innerText !== "--" && isVisible) {
                const probEl = res.probId ? document.getElementById(res.probId) : null;
                const probValue = (probEl && probEl.innerText !== "--") ? ` (Conf: ${probEl.innerText})` : "";
                doc.text(`• ${res.label}: ${el.innerText}${probValue}`, 15, currentY);
                currentY += 5;
            }
        });

        // --- LOCALIZACIÓN YOLO + CONFIDENCE VALUE ---
        const debugEl = document.getElementById('res-error-msg');
        const boxYolo = document.getElementById('boxYolo');

        if (boxYolo && boxYolo.style.display !== 'none' && debugEl) {
            const debugText = debugEl.innerText; // "Class:supraspinatus | Conf:0.90"
            
            if (debugText.includes("Class:")) {
                const detClass = debugText.split('Class:')[1].split('|')[0].trim();
                const detConf = debugText.includes("Conf:") ? debugText.split('Conf:')[1].trim() : "";
                
                doc.setFont("helvetica");
                const confDisplay = detConf ? ` (Confidence: ${detConf})` : "";
                doc.text(`• Rotator Cuff Location: ${detClass}${confDisplay}`, 15, currentY);
                doc.setTextColor(0, 0, 0);
                doc.setFont("helvetica", "normal");
                currentY += 5;
            }
        }

        // --- MORPHOLOGY ---
        const calciumBadge = document.getElementById('calciumBadge');
        if (calciumBadge && calciumBadge.innerText.trim() !== "" && document.getElementById('boxCalcium').style.display !== 'none') {
            doc.setFont("helvetica");
            doc.text(`• Calcification morphology: ${calciumBadge.innerText}`, 15, currentY);
        }

        doc.save(`Medical_Report_Shoulder_${Date.now()}.pdf`);

    } catch (err) {
        console.error(err);
        alert("Report generation failed.");
    } finally {
        btn.innerHTML = originalContent;
        btn.disabled = false;
    }
}