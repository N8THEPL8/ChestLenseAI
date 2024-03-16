document.querySelector('#uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    const action = e.submitter.value === 'Prebuilt' ? '/upload' : '/upload-our-model';
    await handleFormSubmit(e, action);
});

async function handleFormSubmit(e, url) {
    const formData = new FormData(e.target);

    const response = await fetch(url, {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    const p_name = document.getElementById('p_name');
    p_name.innerHTML = `${result.Patient_Name}`;

    const p_id = document.getElementById('p_id');
    p_id.innerHTML = `${result.Patient_ID}`;

    const p_sex = document.getElementById('p_sex');
    p_sex.innerHTML = `${result.Patient_Sex}`;

    const p_bod = document.getElementById('p_bod');
    p_bod.innerHTML = `${result.Patient_Birth_Date}`;

    const p_ad = document.getElementById('p_ad');
    p_ad.innerHTML = `${result.Acquisition_Date}`;

    const p_pos = document.getElementById('p_pos');
    p_pos.innerHTML = `${result.View_Position}`;

    const p_orient = document.getElementById('p_orient');
    p_orient.innerHTML = `${result.Patient_Orientation}`;

    const p_age = document.getElementById('p_age');
    p_age.innerHTML = `${result.Patient_Age_at_Time_of_Acquisition}`;

    const ate_actual = document.getElementById('ate_actual');
    ate_actual.innerHTML = `${result.diseasesData[0].prediction}`;

    const car_actual = document.getElementById('car_actual');
    car_actual.innerHTML = `${result.diseasesData[1].prediction}`;

    const con_actual = document.getElementById('con_actual');
    con_actual.innerHTML = `${result.diseasesData[2].prediction}`;

    const ede_actual = document.getElementById('ede_actual');
    ede_actual.innerHTML = `${result.diseasesData[3].prediction}`;

    const nof_actual = document.getElementById('nof_actual');
    nof_actual.innerHTML = `${result.diseasesData[4].prediction}`;

    const eff_actual = document.getElementById('eff_actual');
    eff_actual.innerHTML = `${result.diseasesData[5].prediction}`;

    const response2 = await fetch('/fetchimage');

    const result2 = await response2.json();

    document.getElementById("output").src = `uploads\\${result2.filename}`;

    const dropdown = document.getElementById('imageOptions');
    dropdown.addEventListener('change', function () {
        console.log('clicked')
        const selectedOption = dropdown.options[dropdown.selectedIndex].value;
        const imageElement = document.getElementById('output');

        if (selectedOption === 'option1') {
            document.getElementById("output").src = `uploads\\${result2.filename}`;
        } else if (selectedOption === 'option2') {
            imageElement.src = 'data:image/jpeg;base64,' + result.diseasesData[0].gradCamImage;
        } else if (selectedOption === 'option3') {
            imageElement.src = 'data:image/jpeg;base64,' + result.diseasesData[1].gradCamImage;
        } else if (selectedOption === 'option4') {
            imageElement.src = 'data:image/jpeg;base64,' + result.diseasesData[2].gradCamImage;
        } else if (selectedOption === 'option5') {
            imageElement.src = 'data:image/jpeg;base64,' + result.diseasesData[3].gradCamImage;
        } else if (selectedOption === 'option6') {
            imageElement.src = 'data:image/jpeg;base64,' + result.diseasesData[4].gradCamImage;
        } else if (selectedOption === 'option7') {
            imageElement.src = 'data:image/jpeg;base64,' + result.diseasesData[5].gradCamImage;
        }
    });

    const response3 = await fetch('/deleteimage');
}