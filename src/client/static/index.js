document.querySelector('form').addEventListener('submit', async function (e) {
    e.preventDefault();
    const formData = new FormData(this);

    const response = await fetch('/upload', {
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

    const ate_prob = document.getElementById('ate_prob');
    ate_prob.innerHTML = `${result.Model_Atelectasis}%`;

    const ate_actual = document.getElementById('ate_actual');
    ate_actual.innerHTML = `${result.Actual_Atelectasis}`;

    const car_prob = document.getElementById('car_prob');
    car_prob.innerHTML = `${result.Model_Cardiomegaly}%`;

    const car_actual = document.getElementById('car_actual');
    car_actual.innerHTML = `${result.Actual_Cardiomegaly}`;

    const con_prob = document.getElementById('con_prob');
    con_prob.innerHTML = `${result.Model_Consolidation}%`;

    const con_actual = document.getElementById('con_actual');
    con_actual.innerHTML = `${result.Actual_Consolidation}`;

    const ede_prob = document.getElementById('ede_prob');
    ede_prob.innerHTML = `${result.Model_Edema}%`;

    const ede_actual = document.getElementById('ede_actual');
    ede_actual.innerHTML = `${result.Actual_Edema}`;

    const eff_prob = document.getElementById('eff_prob');
    eff_prob.innerHTML = `${result.Model_Effusion}%`;

    const eff_actual = document.getElementById('eff_actual');
    eff_actual.innerHTML = `${result.Actual_Effusion}`;

    const response2 = await fetch('/fetchimage');

    const result2 = await response2.json();

    console.log(result2.filename);
    document.getElementById("output").src = `uploads\\${result2.filename}`;

    const response3 = await fetch('/deleteimage');
});

var loadFile = function (event) {
    var image = document.getElementById('output');
    image.src = URL.createObjectURL(event.target.files[0]);
};

function eraseText() {
    document.getElementById("output2").value = "";
}