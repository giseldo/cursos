document.addEventListener('DOMContentLoaded', function () {
  const main = document.querySelector('#app');
  if (!main) return;

  const loggedIn = localStorage.getItem('loggedIn');
  if (loggedIn === 'true') return;

  main.style.display = 'none';

  const loginDiv = document.createElement('div');
  loginDiv.style = 'display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;background:#fff;position:fixed;top:0;left:0;width:100vw;z-index:9999;';
  loginDiv.innerHTML = `
    <h2>Login</h2>
    <input id="user" placeholder="Usuário" style="margin-bottom:8px;padding:4px;">
    <input id="pass" type="password" placeholder="Senha" style="margin-bottom:8px;padding:4px;">
    <button id="loginBtn">Entrar</button>
    <div id="loginError" style="color:red;margin-top:8px;"></div>
  `;
  document.body.appendChild(loginDiv);

  document.getElementById('loginBtn').onclick = function () {
    const user = document.getElementById('user').value;
    const pass = document.getElementById('pass').value;
    // Usuário e senha estáticos
    if (user === 'admin' && pass === '1234') {
      localStorage.setItem('loggedIn', 'true');
      loginDiv.remove();
      main.style.display = '';
    } else {
      document.getElementById('loginError').innerText = 'Usuário ou senha inválidos!';
    }
  };
});
