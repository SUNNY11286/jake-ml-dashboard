// function uploadFile() {
//     const fileInput = document.getElementById('csv-file');
//     const file = fileInput.files[0];
//     if (file) {
//         const formData = new FormData();
//         formData.append('file', file);

//         fetch('/upload', {
//             method: 'POST',
//             body: formData
//         })
//         .then(response => response.json())
//         .then(data => {
//             if (data.error) {
//                 alert(data.error);
//             } else {
//                 populateTargetVariables(data.columns);
//                 document.getElementById('target-selection').style.display = 'block';
//                 document.getElementById('model-selection').style.display = 'block';
//             }
//         })
//         .catch(error => console.error('Error:', error));
//     } else {
//         alert('Please select a file');
//     }
// }

// function populateTargetVariables(columns) {
//     const select = document.getElementById('target-variable');
//     select.innerHTML = '';
//     columns.forEach(column => {
//         const option = document.createElement('option');
//         option.value = column;
//         option.textContent = column;
//         select.appendChild(option);
//     });
// }

// function processData() {
//     const target = document.getElementById('target-variable').value;
//     const modelType = document.getElementById('model-type').value;

//     const formData = new FormData();
//     formData.append('target', target);
//     formData.append('model_type', modelType);

//     fetch('/process', {
//         method: 'POST',
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => {
//         if (data.error) {
//             alert(data.error);
//         } else {
//             displayResults(data);
//         }
//     })
//     .catch(error => console.error('Error:', error));
// }

// function displayResults(data) {
//     document.getElementById('rmse').textContent = data.rmse.toFixed(4);
//     document.getElementById('r2').textContent = data.r2.toFixed(4);
//     document.getElementById('accuracy').textContent = data.accuracy.toFixed(2) + '%';
//     document.getElementById('plot').src = 'data:image/png;base64,' + data.plot_url;
//     document.getElementById('results').style.display = 'block';
// }





function validateForm() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    if (!email || !password) {
        alert('Please fill in all fields.');
        return false;
    }
    // You can add more validation logic if required (e.g., regex for email)
    return true;
}
