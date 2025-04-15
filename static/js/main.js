// static/js/main.js
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const processingIndicator = document.getElementById('processing-indicator');
    const resultsDiv = document.getElementById('results');
    const processAnotherBtn = document.getElementById('process-another');
    
    // Result elements
    const resultDate = document.getElementById('result-date');
    const resultVendor = document.getElementById('result-vendor');
    const resultAmount = document.getElementById('result-amount');
    const resultItems = document.getElementById('result-items');
    const resultRawText = document.getElementById('result-raw-text');
    
    // Click on upload area to trigger file input
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Handle file selection
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            handleFile(this.files[0]);
        }
    });
    
    // Handle drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    // Process another document
    processAnotherBtn.addEventListener('click', function() {
        resultsDiv.style.display = 'none';
        uploadArea.style.display = 'block';
        fileInput.value = '';
    });
    
    // Handle file processing
    function handleFile(file) {
        // Check file type
        const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'application/pdf'];
        if (!validTypes.includes(file.type)) {
            showError('Please upload an image (JPG, PNG, BMP, TIFF) or PDF file.');
            return;
        }
        
        // Show processing indicator
        uploadArea.style.display = 'none';
        processingIndicator.style.display = 'block';
        resultsDiv.style.display = 'none';
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Send to backend
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Error processing document');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide processing indicator
            processingIndicator.style.display = 'none';
            
            // Display results
            displayResults(data);
            
            // Show results section
            resultsDiv.style.display = 'block';
        })
        .catch(error => {
            processingIndicator.style.display = 'none';
            uploadArea.style.display = 'block';
            showError(error.message);
        });
    }
    
    function displayResults(data) {
        // Display date
        resultDate.textContent = data.date || 'Not found';
        
        // Display vendor
        resultVendor.textContent = data.vendor || 'Not found';
        
        // Display amount
        resultAmount.textContent = data.total_amount || 'Not found';
        
        // Display line items
        if (data.items && data.items.length > 0) {
            resultItems.innerHTML = '';
            data.items.forEach(item => {
                const li = document.createElement('li');
                li.textContent = `${item.name} — Qty: ${item.quantity || 'N/A'}, Unit Price: ${item.unit_price || 'N/A'}, Total: ${item.total || 'N/A'}`;
                resultItems.appendChild(li);
            });
        } else {
            resultItems.innerHTML = '<li>No items found</li>';
        }
        
        // Display raw text
        resultRawText.textContent = data.raw_text || 'No text extracted';
    }
    
    function showError(message) {
        // Remove any existing error
        const existingError = document.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }
        
        // Create error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        // Insert after upload area
        uploadArea.parentNode.insertBefore(errorDiv, uploadArea.nextSibling);
        
        // Remove after 5 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
});