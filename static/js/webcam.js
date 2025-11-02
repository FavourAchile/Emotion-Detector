(async ()=>{
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const snapBtn = document.getElementById('snap');
  const ctx = canvas.getContext('2d');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
  } catch (e) {
    alert('Allow camera access to use webcam capture');
    console.error(e);
  }

  snapBtn.addEventListener('click', async ()=>{
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/png');
    const name = document.getElementById('nameCam').value;
    const res = await fetch('/predict_camera', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ name: name, image: dataUrl })
    });
    const json = await res.json();
    document.getElementById('cameraResult').innerText = JSON.stringify(json, null, 2);
  });
})();
