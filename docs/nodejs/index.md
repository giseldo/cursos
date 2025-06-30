# Aula 01: Introdução ao Node.js

:::info
Criado por Kleber Galvão.
:::

Bem-vindo à primeira aula do curso **Node.js do Básico ao Avançado**! Esta aula é projetada para fornecer uma base sólida sobre o que é o Node.js, como ele funciona internamente, seus principais casos de uso e como começar a utilizá-lo com um exemplo prático. Vamos mergulhar em uma explicação detalhada e prática, cobrindo teoria, exemplos e configurações, garantindo que você compreenda os conceitos fundamentais e esteja preparado para os próximos passos do curso.

O objetivo desta aula é que você entenda o que torna o Node.js único, como sua arquitetura não-bloqueante e o Event Loop funcionam, e como configurar um ambiente de desenvolvimento funcional. Além disso, você criará seu primeiro servidor "Hello World" usando o módulo nativo `http` do Node.js. O material é estruturado para ser claro, abrangente e prático, com exemplos comentados e explicações detalhadas.

---

## Teoria: O que é Node.js?

### 1. Introdução ao Node.js

Node.js é uma **plataforma de desenvolvimento** que permite executar JavaScript fora do navegador. Ele foi criado em 2009 por Ryan Dahl e é baseado no motor **V8** do Google Chrome, que compila e executa código JavaScript com alta eficiência. Diferentemente do JavaScript tradicional, que roda no lado do cliente (navegadores), o Node.js possibilita o uso de JavaScript no lado do servidor, permitindo a construção de aplicações backend, como APIs, servidores web, ferramentas de linha de comando (CLI) e sistemas em tempo real.

Node.js é amplamente adotado por empresas como **Netflix**, **Uber**, **LinkedIn** e **PayPal** devido à sua alta performance, escalabilidade e capacidade de lidar com operações assíncronas. Ele é particularmente adequado para aplicações que requerem baixa latência e alta concorrência, como chats em tempo real, streaming de vídeo e APIs RESTful.

#### Por que Node.js é diferente?

- **JavaScript no servidor**: Antes do Node.js, JavaScript era limitado a navegadores. O Node.js trouxe a linguagem para o backend, permitindo que desenvolvedores usem uma única linguagem para o frontend e o backend, simplificando o desenvolvimento full-stack.
- **Leve e rápido**: O motor V8 é altamente otimizado, garantindo execução rápida de código JavaScript.
- **Ecossistema npm**: O Node.js vem com o **npm** (Node Package Manager), que abriga milhões de pacotes open-source, permitindo que desenvolvedores adicionem funcionalidades rapidamente.
- **Comunidade ativa**: A comunidade Node.js é uma das maiores do mundo, com constante atualização de bibliotecas, ferramentas e suporte em plataformas como X (#NodeJS) e Stack Overflow.

#### Histórico e Contexto

O Node.js surgiu em um momento em que o desenvolvimento web exigia soluções mais escaláveis para lidar com o crescimento de aplicações em tempo real. Antes de 2009, servidores web tradicionais (como Apache com PHP) usavam modelos síncronos baseados em threads, onde cada conexão de cliente consumia uma thread, limitando a escalabilidade em cenários de alta concorrência. Ryan Dahl percebeu que o JavaScript, com sua natureza assíncrona (ex.: callbacks em eventos DOM), poderia ser aproveitado para criar um servidor mais eficiente. Assim, o Node.js foi projetado para maximizar a concorrência usando um modelo não-bloqueante.

### 2. Arquitetura do Node.js: Event Loop e Não-Bloqueante

O diferencial do Node.js está em sua **arquitetura assíncrona e não-bloqueante**, impulsionada pelo **Event Loop**. Para entender como isso funciona, vamos explorar os conceitos fundamentais.

#### 2.1. Modelo Não-Bloqueante

Em linguagens tradicionais como PHP ou Ruby (em configurações padrão), as operações de entrada/saída (I/O), como leitura de arquivos, consultas a bancos de dados ou chamadas de rede, são síncronas por padrão. Isso significa que o programa "para" enquanto espera a conclusão de uma operação. Por exemplo, ao ler um arquivo, o servidor aguarda até que o arquivo esteja completamente carregado antes de processar a próxima tarefa.

No Node.js, as operações de I/O são **assíncronas** e **não-bloqueantes**. O Node.js não espera a conclusão de uma tarefa de I/O para prosseguir. Em vez disso, ele delega a tarefa ao sistema operacional e continua executando outras partes do código. Quando a tarefa de I/O é concluída, o Node.js é notificado por meio de **callbacks**, **promises** ou **async/await**.

**Exemplo ilustrativo**:

Imagine um restaurante onde o garçom (o Node.js) recebe pedidos de várias mesas (tarefas). Em um modelo síncrono, o garçom ficaria parado na cozinha esperando cada prato ficar pronto antes de atender outra mesa. No modelo não-bloqueante do Node.js, o garçom entrega o pedido à cozinha e imediatamente atende outras mesas, voltando apenas quando o prato está pronto.

Essa abordagem permite que o Node.js lide com **milhares de conexões simultâneas** com eficiência, tornando-o ideal para aplicações que requerem alta concorrência, como servidores de chat ou streaming.

#### 2.2. O Event Loop

O **Event Loop** é o mecanismo central do Node.js, responsável por gerenciar todas as operações assíncronas. Ele opera em uma **única thread**, mas delega operações de I/O (como leitura de arquivos ou chamadas HTTP) ao sistema operacional, que as executa em threads separadas no background (via **libuv**, a biblioteca de I/O do Node.js). Quando uma tarefa de I/O é concluída, o Event Loop é notificado e executa a callback associada.

##### Como o Event Loop funciona?

O Event Loop é um loop contínuo que verifica se há tarefas pendentes em diferentes **fases**. Cada fase é responsável por um tipo específico de tarefa. As principais fases são:

1. **Timers**: Executa callbacks de `setTimeout` e `setInterval` que atingiram seu tempo limite.
2. **Pending Callbacks**: Executa callbacks de operações de I/O que foram concluídas (ex.: leitura de arquivos).
3. **Idle, Prepare**: Fases internas usadas pelo Node.js para manutenção.
4. **Poll**: Recupera novos eventos de I/O (ex.: conexões de rede ou leitura de arquivos). Se não houver eventos, o Event Loop pode pausar aqui.
5. **Check**: Executa callbacks de `setImmediate`.
6. **Close Callbacks**: Executa callbacks de eventos de fechamento (ex.: fechar uma conexão de socket).

**Exemplo visual**:

Imagine uma roda-giratória em um parque de diversões. A roda (Event Loop) gira continuamente, verificando cada "assento" (fase) para ver se há uma tarefa pronta para ser executada. Se não houver tarefas, a roda continua girando. Se uma tarefa de I/O (como ler um arquivo) é iniciada, ela é delegada ao sistema operacional, e o Event Loop continua girando, verificando outras tarefas. Quando o arquivo está pronto, a callback associada é colocada na fila para ser executada na próxima iteração.

##### Fluxo Simplificado do Event Loop

1. O Node.js inicia e entra no Event Loop.
2. Uma requisição HTTP chega ao servidor.
3. O Node.js delega a tarefa (ex.: buscar dados de um banco) ao sistema operacional via libuv.
4. Enquanto espera, o Event Loop processa outras requisições ou tarefas.
5. Quando os dados do banco estão prontos, a callback é enfileirada.
6. O Event Loop executa a callback na fase apropriada, enviando a resposta ao cliente.

#### 2.3. Benefícios da Arquitetura

- **Alta concorrência**: O Node.js pode lidar com milhares de conexões simultâneas com uma única thread, ao contrário de servidores tradicionais que criam uma thread por conexão.
- **Baixa latência**: Como as operações não bloqueiam, o Node.js responde rapidamente, mesmo sob alta carga.
- **Escalabilidade**: Ideal para aplicações que precisam escalar horizontalmente, como APIs e sistemas em tempo real.
- **Eficiência de recursos**: Consome menos memória e CPU em comparação com modelos baseados em threads.

#### 2.4. Limitações

Embora o Node.js seja poderoso, ele não é ideal para todas as situações:

- **Tarefas intensivas de CPU**: Como o Node.js opera em uma única thread, tarefas que exigem muito processamento (ex.: cálculos complexos ou compressão de vídeo) podem bloquear o Event Loop, reduzindo a performance. Para isso, o Node.js oferece o módulo `worker_threads`, que será abordado em módulos avançados.
- **Debugging complexo**: A natureza assíncrona pode dificultar a depuração em projetos mal estruturados, especialmente para iniciantes.
- **Curva de aprendizado**: Entender o Event Loop e gerenciar callbacks ou promises requer prática.

### 3. Casos de Uso do Node.js

Node.js é versátil e usado em uma ampla gama de aplicações. Aqui estão os principais casos de uso, com exemplos reais:

#### 3.1. APIs RESTful

Node.js é amplamente utilizado para criar **APIs RESTful** devido à sua capacidade de lidar com muitas requisições simultâneas. Frameworks como **Express** simplificam a criação de rotas, middlewares e integração com bancos de dados.

**Exemplo real**: A **Netflix** usa Node.js para construir APIs que entregam conteúdo personalizado aos usuários, lidando com milhões de requisições por segundo.

#### 3.2. Aplicações em Tempo Real

Aplicações que exigem comunicação em tempo real, como chats, jogos online ou notificações ao vivo, se beneficiam da arquitetura não-bloqueante do Node.js. Bibliotecas como **Socket.IO** facilitam a implementação de WebSockets.

**Exemplo real**: O **Slack** usa Node.js para suportar mensagens em tempo real e notificações push.

#### 3.3. Streaming de Dados

Node.js é ideal para **streaming de dados**, como vídeos ou arquivos grandes, pois pode processar dados em pedaços (chunks) sem carregar tudo na memória.

**Exemplo real**: A **Netflix** utiliza Node.js para streaming de vídeo, otimizando a entrega de conteúdo em alta escala.

#### 3.4. Ferramentas CLI

Node.js é amplamente usado para criar **ferramentas de linha de comando**, como o `create-react-app` ou o `vue-cli`, devido à sua facilidade de manipulação de arquivos e processos.

**Exemplo real**: O **npm**, o gerenciador de pacotes do Node.js, é ele próprio uma ferramenta CLI escrita em Node.js.

#### 3.5. Microserviços

A arquitetura leve do Node.js o torna ideal para **microserviços**, onde diferentes serviços podem ser escritos e escalados independentemente.

**Exemplo real**: A **Uber** usa Node.js em sua arquitetura de microserviços para gerenciar diferentes partes de sua plataforma, como localização de motoristas e pagamentos.

#### 3.6. Internet das Coisas (IoT)

Node.js é usado em dispositivos IoT devido à sua leveza e capacidade de lidar com muitas conexões simultâneas.

**Exemplo real**: Empresas de automação residencial usam Node.js para gerenciar dispositivos conectados, como lâmpadas inteligentes e sensores.

### 4. Por que Aprender Node.js em 2025?

- **Demanda de mercado**: Node.js é uma das tecnologias mais requisitadas em vagas de desenvolvimento backend e full-stack, segundo plataformas como LinkedIn e Indeed.
- **Ecossistema maduro**: Com frameworks como Express, NestJS e Fastify, o Node.js é robusto e moderno.
- **Integração com tecnologias modernas**: Node.js se integra facilmente com **TypeScript**, bancos NoSQL (MongoDB), bancos relacionais (PostgreSQL) e ferramentas de CI/CD.
- **Comunidade e recursos**: Milhares de pacotes no npm, tutoriais, e suporte ativo em comunidades como X (#NodeJS), Reddit (r/node) e GitHub.
- **Versatilidade**: Node.js é usado em startups, grandes empresas e projetos open-source, oferecendo oportunidades em diversos setores.

---

## Ferramentas: Configurando o Ambiente

Antes de mergulharmos no exemplo prático, vamos configurar o ambiente de desenvolvimento. Esta seção cobre a instalação do Node.js, npm e a configuração do Visual Studio Code (VS Code) para garantir que você tenha tudo pronto para começar.

### 1. Instalando o Node.js e npm

O Node.js inclui o **npm** (Node Package Manager) por padrão, que será usado para gerenciar dependências e scripts.

#### Passos para Instalação

1. **Baixe o Node.js**:
   - Acesse o site oficial: [nodejs.org](https://nodejs.org).
   - Baixe a versão **LTS** (Long Term Support), que é a mais estável. Em 2025, a versão LTS mais recente provavelmente será a v20.x ou superior.
   - Para **Windows** e **macOS**, use o instalador oficial. Para **Linux**, siga as instruções específicas para sua distribuição:
     ```bash
     # Exemplo para Ubuntu
     sudo apt update
     sudo apt install nodejs npm
     ```
2. **Verifique a Instalação**:
   - Abra o terminal (Prompt de Comando, PowerShell ou Bash) e execute:
     ```bash
     node -v
     npm -v
     ```
   - Isso deve retornar as versões instaladas (ex.: `v20.12.2` e `10.5.0`).
3. **Atualize o npm (opcional)**:
   - Para garantir a versão mais recente do npm, execute:
     ```bash
     npm install -g npm@latest
     ```

#### Dica Avançada

Considere usar o **nvm** (Node Version Manager) para gerenciar múltiplas versões do Node.js. Isso é útil para projetos que exigem versões específicas:

```bash
# Instalação do nvm (Linux/macOS)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
# Instalar uma versão específica
nvm install 20
```

### 2. Configurando o Visual Studio Code

O **VS Code** é um editor de código leve e poderoso, amplamente usado para desenvolvimento Node.js.

#### Passos para Configuração

1. **Instale o VS Code**:
   - Baixe e instale em [code.visualstudio.com](https://code.visualstudio.com).
2. **Instale Extensões Úteis**:
   - **ESLint**: Para linting e formatação de código JavaScript.
   - **Prettier**: Para formatação automática de código.
   - **Node.js Extension Pack**: Inclui suporte para depuração e snippets.
   - **REST Client**: Para testar APIs diretamente no VS Code.
   - Para instalar, abra o VS Code, vá para a aba de extensões (`Ctrl+Shift+X`) e pesquise pelos nomes.
3. **Configurar o Terminal Integrado**:
   - Abra o terminal integrado (`Ctrl+``) e verifique se o Node.js está acessível com `node -v`.
4. **Configurar Depuração**:
   - Crie um arquivo `launch.json` no diretório `.vscode`:
     ```json
     {
       "version": "0.2.0",
       "configurations": [
         {
           "type": "node",
           "request": "launch",
           "name": "Launch Program",
           "program": "${workspaceFolder}/index.js"
         }
       ]
     }
     ```

### 3. Estrutura do Projeto

Crie uma pasta para seus projetos Node.js:

```bash
mkdir meu-primeiro-projeto
cd meu-primeiro-projeto
npm init -y
```

Isso cria um arquivo `package.json` com configurações padrão:

```json
{
  "name": "meu-primeiro-projeto",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [],
  "author": "",
  "license": "ISC"
}
```

O `package.json` é o coração de qualquer projeto Node.js, definindo dependências, scripts e metadados.

---

## Exemplo Prático: Criando um "Hello World" com o Módulo `http`

Agora que o ambiente está configurado, vamos criar nosso primeiro servidor Node.js usando o módulo nativo `http`. Este exemplo é simples, mas demonstra os conceitos fundamentais de criação de um servidor web com Node.js.

### Objetivo do Exemplo

Criar um servidor HTTP que escuta na porta 3000 e responde com a mensagem "Hello World" quando acessado via navegador ou ferramenta como Postman.

### Passo 1: Criar o Arquivo Principal

1. Na pasta `meu-primeiro-projeto`, crie um arquivo chamado `index.js`.
2. Adicione o seguinte código:

```javascript
// Importa o módulo http nativo do Node.js
const http = require('http');

// Define o hostname e a porta onde o servidor vai rodar
const hostname = '127.0.0.1'; // localhost
const port = 3000;

// Cria o servidor HTTP
const server = http.createServer((req, res) => {
  // Define o status da resposta como 200 (OK) e o tipo de conteúdo como texto simples
  res.statusCode = 200;
  res.setHeader('Content-Type', 'text/plain');
  
  // Envia a mensagem "Hello World" como resposta
  res.end('Hello World\n');
});

// Inicia o servidor e escuta na porta especificada
server.listen(port, hostname, () => {
  console.log(`Servidor rodando em http://${hostname}:${port}/`);
});
```

### Explicação do Código

- **`const http = require('http')`**: Importa o módulo `http`, que permite criar servidores e clientes HTTP. O `require` é o sistema de módulos CommonJS, usado por padrão no Node.js.
- **`http.createServer`**: Cria um servidor HTTP. A função `(req, res)` é chamada toda vez que uma requisição HTTP é recebida:
  - `req` (request): Contém informações sobre a requisição, como método (GET, POST), URL e headers.
  - `res` (response): Usado para enviar a resposta ao cliente, como status, headers e corpo.
- **`res.statusCode = 200`**: Define o código de status HTTP (200 significa "OK").
- **`res.setHeader`**: Define o tipo de conteúdo da resposta como `text/plain`.
- **`res.end`**: Envia a resposta ao cliente e finaliza a conexão.
- **`server.listen`**: Inicia o servidor na porta e hostname especificados, exibindo uma mensagem no console quando o servidor está ativo.

### Passo 2: Executar o Servidor

1. No terminal, na pasta do projeto, execute:
   ```bash
   node index.js
   ```
2. Você verá a mensagem: `Servidor rodando em http://127.0.0.1:3000/`.
3. Abra um navegador e acesse `http://localhost:3000`. Você verá a mensagem "Hello World".
4. Para parar o servidor, pressione `Ctrl+C` no terminal.

### Passo 3: Testar com Ferramentas

- Use o **Postman** ou **curl** para testar a requisição:
  ```bash
  curl http://localhost:3000
  ```
  Isso deve retornar `Hello World`.

### Passo 4: Explorando Mais

Vamos expandir o exemplo para lidar com diferentes rotas. Modifique o `index.js`:

```javascript
const http = require('http');

const hostname = '127.0.0.1';
const port = 3000;

const server = http.createServer((req, res) => {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'text/plain');

  // Verifica a URL da requisição
  if (req.url === '/') {
    res.end('Bem-vindo à página inicial!\n');
  } else if (req.url === '/about') {
    res.end('Esta é a página Sobre!\n');
  } else if (req.url === '/contact') {
    res.end('Entre em contato conosco!\n');
  } else {
    res.statusCode = 404;
    res.end('Página não encontrada!\n');
  }
});

server.listen(port, hostname, () => {
  console.log(`Servidor rodando em http://${hostname}:${port}/`);
});
```

#### Explicação do Código Expandido

- **`req.url`**: Contém o caminho da URL solicitada (ex.: `/`, `/about`).
- **Condicional `if/else`**: Verifica a URL e retorna uma resposta específica.
- **Status 404**: Retornado quando a URL não corresponde a nenhuma rota conhecida.

Teste as rotas acessando:

- `http://localhost:3000/` → "Bem-vindo à página inicial!"
- `http://localhost:3000/about` → "Esta é a página Sobre!"
- `http://localhost:3000/contact` → "Entre em contato conosco!"
- `http://localhost:3000/qualquercoisa` → "Página não encontrada!"

### Passo 5: Depuração

Se algo der errado (ex.: porta já em uso), você verá um erro no terminal. Para depurar:

- Verifique se a porta 3000 está livre:
  ```bash
  # Windows
  netstat -a -n -o | find "3000"
  # Linux/macOS
  lsof -i :3000
  ```
- Use o VS Code para depuração:
  - Coloque um breakpoint clicando à esquerda de uma linha no `index.js`.
  - Pressione `F5` para iniciar a depuração e inspecionar variáveis como `req.url`.

---

## Boas Práticas e Dicas

1. **Organize seu Código**:
   - Mantenha o código claro e comentado, mesmo em exemplos simples.
   - Use constantes para valores fixos, como `hostname` e `port`.

2. **Evite Bloqueios**:
   - O módulo `http` é assíncrono por natureza, mas evite operações síncronas (ex.: `fs.readFileSync`) em servidores reais, pois elas bloqueiam o Event Loop.

3. **Versionamento**:
   - Inicialize um repositório Git:
     ```bash
     git init
     git add .
     git commit -m "Primeiro servidor Node.js"
     ```

4. **Teste Incrementalmente**:
   - Faça pequenas alterações e teste frequentemente para evitar erros acumulados.

5. **Explore o npm**:
   - Instale o `nodemon` para reiniciar o servidor automaticamente ao salvar alterações:
     ```bash
     npm install --save-dev nodemon
     ```
     Adicione ao `package.json`:
     ```json
     "scripts": {
       "start": "node index.js",
       "dev": "nodemon index.js"
     }
     ```
     Execute com `npm run dev`.

---

## Conclusão

Nesta aula, você aprendeu:

- **O que é Node.js**: Uma plataforma para executar JavaScript no servidor, baseada no motor V8.
- **Event Loop e Arquitetura Não-Bloqueante**: Como o Node.js lida com operações assíncronas para alta concorrência.
- **Casos de Uso**: APIs, streaming, aplicações em tempo real, CLIs e microserviços.
- **Configuração do Ambiente**: Instalação do Node.js, npm e VS Code.
- **Exemplo Prático**: Criação de um servidor HTTP simples com o módulo `http`.

O exemplo "Hello World" é apenas o começo, mas demonstra como o Node.js pode criar servidores web rapidamente e introduz conceitos fundamentais como requisições, respostas e o Event Loop. Nos próximos módulos, você expandirá esse conhecimento para criar APIs robustas, manipular arquivos e integrar bancos de dados.

### Próximos Passos

- Experimente adicionar mais rotas ao servidor e testar diferentes métodos HTTP (ex.: POST).
- Explore a documentação do módulo `http` em [nodejs.org](https://nodejs.org/api/http.html).
- Prepare-se para o próximo módulo, onde abordaremos **Módulos e CommonJS vs. ES Modules**.

Se tiver dúvidas ou quiser aprofundar algum tópico, é só perguntar! 🚀
