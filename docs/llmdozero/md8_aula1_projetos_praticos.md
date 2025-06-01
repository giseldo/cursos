# Projetos Práticos e Aplicações Avançadas

Neste módulo final, vamos aplicar todo o conhecimento adquirido ao longo do curso para desenvolver projetos práticos completos. Aprenderemos a criar aplicações reais que utilizam LLMs, desde chatbots até assistentes especializados, tudo otimizado para hardware com limitações de memória.

## Criando um Chatbot com Memória de Contexto

Vamos começar implementando um chatbot que mantém memória de contexto, mesmo com recursos limitados.

### Arquitetura do Chatbot

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
import time

class MemoryEfficientChatbot:
    """
    Chatbot que mantém memória de contexto otimizado para hardware limitado.
    """
    def __init__(
        self,
        model_name: str,
        max_context_length: int = 1024,
        sliding_window: int = 512,
        memory_limit_mb: Optional[int] = None,
        use_8bit: bool = False,
        use_4bit: bool = False,
        device: Optional[str] = None
    ):
        """
        Inicializa o chatbot.
        
        Args:
            model_name: Nome ou caminho do modelo
            max_context_length: Comprimento máximo do contexto
            sliding_window: Tamanho da janela deslizante para contexto longo
            memory_limit_mb: Limite de memória em MB (None para sem limite)
            use_8bit: Se True, usa quantização de 8 bits
            use_4bit: Se True, usa quantização de 4 bits (tem precedência sobre 8 bits)
            device: Dispositivo para execução ("cpu", "cuda", "auto")
        """
        self.max_context_length = max_context_length
        self.sliding_window = sliding_window
        self.memory_limit_mb = memory_limit_mb
        
        # Determinar dispositivo
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Carregar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Garantir que o tokenizer tenha tokens especiais
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configurar quantização se solicitado
        if use_4bit or use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                
                if use_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                else:  # use_8bit
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                
                # Carregar modelo quantizado
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device.type == "cuda" else {"": "cpu"}
                )
                
                print(f"Modelo carregado com quantização de {4 if use_4bit else 8} bits")
            except ImportError:
                print("Biblioteca bitsandbytes não disponível, carregando modelo sem quantização")
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.to(self.device)
        else:
            # Carregar modelo sem quantização
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
        
        # Colocar modelo em modo de avaliação
        self.model.eval()
        
        # Inicializar histórico de conversa
        self.conversation_history = []
        
        # Definir templates de mensagem
        self.system_template = "Você é um assistente útil e amigável."
        self.user_template = "Usuário: {message}"
        self.assistant_template = "Assistente: {message}"
        
        # Verificar memória disponível
        if self.memory_limit_mb is not None and torch.cuda.is_available():
            self.check_memory_usage()
    
    def check_memory_usage(self):
        """
        Verifica o uso de memória da GPU.
        
        Returns:
            float: Memória usada em MB
        """
        if not torch.cuda.is_available():
            return 0
        
        # Obter memória alocada
        memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
        print(f"Memória GPU alocada: {memory_allocated:.2f} MB")
        
        # Verificar se excede o limite
        if self.memory_limit_mb is not None and memory_allocated > self.memory_limit_mb:
            print(f"Aviso: Uso de memória ({memory_allocated:.2f} MB) excede o limite ({self.memory_limit_mb} MB)")
            print("Realizando limpeza de memória...")
            
            # Limpar cache CUDA
            torch.cuda.empty_cache()
            
            # Reduzir histórico de conversa se necessário
            if len(self.conversation_history) > 2:
                # Manter apenas as últimas mensagens
                self.conversation_history = self.conversation_history[-2:]
                print("Histórico de conversa reduzido para economizar memória")
        
        return memory_allocated
    
    def set_system_message(self, message: str):
        """
        Define a mensagem do sistema.
        
        Args:
            message: Nova mensagem do sistema
        """
        self.system_template = message
    
    def add_message(self, role: str, content: str):
        """
        Adiciona uma mensagem ao histórico de conversa.
        
        Args:
            role: Papel do remetente ("user" ou "assistant")
            content: Conteúdo da mensagem
        """
        if role == "user":
            formatted_message = self.user_template.format(message=content)
        elif role == "assistant":
            formatted_message = self.assistant_template.format(message=content)
        else:
            formatted_message = content
        
        self.conversation_history.append({"role": role, "content": formatted_message})
    
    def _build_prompt(self) -> str:
        """
        Constrói o prompt completo a partir do histórico de conversa.
        
        Returns:
            str: Prompt completo
        """
        # Começar com a mensagem do sistema
        prompt = self.system_template + "\n\n"
        
        # Adicionar histórico de conversa
        for message in self.conversation_history:
            prompt += message["content"] + "\n"
        
        # Adicionar prefixo para resposta do assistente
        prompt += "Assistente: "
        
        return prompt
    
    def _truncate_conversation_if_needed(self, max_tokens: int):
        """
        Trunca o histórico de conversa se exceder o limite de tokens.
        
        Args:
            max_tokens: Número máximo de tokens permitidos
        """
        # Construir prompt completo
        prompt = self._build_prompt()
        
        # Tokenizar prompt
        tokens = self.tokenizer.encode(prompt)
        
        # Verificar se excede o limite
        if len(tokens) > max_tokens:
            print(f"Aviso: Prompt excede o limite de tokens ({len(tokens)} > {max_tokens})")
            
            # Remover mensagens mais antigas até ficar dentro do limite
            while len(tokens) > max_tokens and len(self.conversation_history) > 1:
                # Remover a mensagem mais antiga
                self.conversation_history.pop(0)
                
                # Reconstruir prompt e tokenizar
                prompt = self._build_prompt()
                tokens = self.tokenizer.encode(prompt)
            
            print(f"Histórico truncado para {len(tokens)} tokens")
    
    def generate_response(
        self,
        user_message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> str:
        """
        Gera uma resposta para a mensagem do usuário.
        
        Args:
            user_message: Mensagem do usuário
            max_new_tokens: Número máximo de tokens a gerar
            temperature: Temperatura para amostragem
            top_p: Probabilidade cumulativa para nucleus sampling
            top_k: Número de tokens mais prováveis a considerar
            repetition_penalty: Penalidade para repetição de tokens
            do_sample: Se True, amostra da distribuição, caso contrário usa greedy decoding
            
        Returns:
            str: Resposta gerada
        """
        # Adicionar mensagem do usuário ao histórico
        self.add_message("user", user_message)
        
        # Truncar conversa se necessário
        self._truncate_conversation_if_needed(self.max_context_length - max_new_tokens)
        
        # Construir prompt
        prompt = self._build_prompt()
        
        # Tokenizar prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Gerar resposta
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decodificar resposta
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrair apenas a parte gerada
        assistant_response = full_response[len(prompt):]
        
        # Limpar resposta (remover texto após possível nova mensagem do usuário)
        if "Usuário:" in assistant_response:
            assistant_response = assistant_response.split("Usuário:")[0].strip()
        
        # Adicionar resposta ao histórico
        self.add_message("assistant", assistant_response)
        
        # Verificar uso de memória
        if self.memory_limit_mb is not None:
            self.check_memory_usage()
        
        # Limpar cache CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return assistant_response
    
    def chat(self):
        """
        Inicia uma sessão de chat interativa no console.
        """
        print("Chatbot inicializado. Digite 'sair' para encerrar.")
        print(f"Usando dispositivo: {self.device}")
        
        while True:
            user_input = input("\nVocê: ")
            
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Encerrando chat...")
                break
            
            start_time = time.time()
            response = self.generate_response(user_input)
            end_time = time.time()
            
            print(f"\nAssistente: {response}")
            print(f"[Tempo de resposta: {end_time - start_time:.2f}s]")
    
    def clear_history(self):
        """
        Limpa o histórico de conversa.
        """
        self.conversation_history = []
        
        # Limpar cache CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### Exemplo de Uso do Chatbot

```python
def chatbot_example():
    """
    Exemplo de uso do chatbot.
    """
    # Inicializar chatbot com modelo pequeno
    chatbot = MemoryEfficientChatbot(
        model_name="distilgpt2",  # Modelo pequeno para exemplo
        max_context_length=512,
        use_8bit=True,            # Usar quantização de 8 bits
        memory_limit_mb=1000      # Limitar uso de memória a 1GB
    )
    
    # Definir mensagem do sistema
    chatbot.set_system_message("Você é um assistente especializado em Python e desenvolvimento de LLMs.")
    
    # Exemplo de conversa
    responses = []
    
    # Primeira mensagem
    response1 = chatbot.generate_response(
        "Como posso implementar um tokenizer simples para processamento de texto em Python?"
    )
    responses.append(response1)
    
    # Segunda mensagem (com contexto da primeira)
    response2 = chatbot.generate_response(
        "Quais bibliotecas posso usar para tornar esse tokenizer mais eficiente?"
    )
    responses.append(response2)
    
    # Terceira mensagem (mudando de assunto)
    response3 = chatbot.generate_response(
        "Explique como funciona o mecanismo de atenção em um modelo Transformer."
    )
    responses.append(response3)
    
    return responses

def run_interactive_chatbot():
    """
    Executa o chatbot em modo interativo.
    """
    # Verificar GPU disponível
    if torch.cuda.is_available():
        print(f"GPU disponível: {torch.cuda.get_device_name(0)}")
        print(f"Memória GPU total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("GPU não disponível, usando CPU")
    
    # Perguntar ao usuário qual modelo usar
    print("\nEscolha um modelo para o chatbot:")
    print("1. distilgpt2 (pequeno, rápido)")
    print("2. gpt2-medium (médio, melhor qualidade)")
    print("3. EleutherAI/pythia-1.4b (maior, mais lento)")
    
    choice = input("Digite o número da opção (padrão: 1): ").strip()
    
    if choice == "2":
        model_name = "gpt2-medium"
    elif choice == "3":
        model_name = "EleutherAI/pythia-1.4b"
    else:
        model_name = "distilgpt2"
    
    # Perguntar sobre quantização
    use_quantization = input("Usar quantização para economizar memória? (s/n, padrão: s): ").strip().lower()
    use_8bit = use_quantization != "n"
    
    # Inicializar chatbot
    print(f"\nInicializando chatbot com modelo {model_name}...")
    chatbot = MemoryEfficientChatbot(
        model_name=model_name,
        max_context_length=1024,
        use_8bit=use_8bit,
        memory_limit_mb=2000  # 2GB
    )
    
    # Definir personalidade
    personality = input("\nDefina a personalidade do chatbot (deixe em branco para padrão): ").strip()
    if personality:
        chatbot.set_system_message(personality)
    
    # Iniciar chat
    chatbot.chat()
```

## Assistente de Programação Especializado

Vamos criar um assistente de programação especializado em Python, otimizado para hardware limitado.

### Implementação do Assistente

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os
from typing import List, Dict, Optional, Union, Tuple

class PythonCodingAssistant:
    """
    Assistente de programação especializado em Python.
    """
    def __init__(
        self,
        model_name: str,
        use_8bit: bool = False,
        use_4bit: bool = False,
        device: Optional[str] = None,
        code_generation_config: Optional[Dict] = None
    ):
        """
        Inicializa o assistente de programação.
        
        Args:
            model_name: Nome ou caminho do modelo
            use_8bit: Se True, usa quantização de 8 bits
            use_4bit: Se True, usa quantização de 4 bits
            device: Dispositivo para execução ("cpu", "cuda", "auto")
            code_generation_config: Configuração para geração de código
        """
        # Determinar dispositivo
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Carregar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Garantir que o tokenizer tenha tokens especiais
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configurar quantização se solicitado
        if use_4bit or use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                
                if use_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                else:  # use_8bit
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                
                # Carregar modelo quantizado
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device.type == "cuda" else {"": "cpu"}
                )
                
                print(f"Modelo carregado com quantização de {4 if use_4bit else 8} bits")
            except ImportError:
                print("Biblioteca bitsandbytes não disponível, carregando modelo sem quantização")
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.to(self.device)
        else:
            # Carregar modelo sem quantização
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
        
        # Colocar modelo em modo de avaliação
        self.model.eval()
        
        # Configuração para geração de código
        self.code_generation_config = {
            "temperature": 0.2,           # Temperatura mais baixa para código mais determinístico
            "top_p": 0.95,                # Valor alto para manter diversidade
            "top_k": 50,                  # Limitar aos 50 tokens mais prováveis
            "repetition_penalty": 1.2,    # Penalidade para evitar repetições
            "max_new_tokens": 512,        # Limite de tokens para geração
            "do_sample": True             # Usar amostragem
        }
        
        # Sobrescrever com configuração personalizada se fornecida
        if code_generation_config:
            self.code_generation_config.update(code_generation_config)
        
        # Templates para diferentes tarefas
        self.templates = {
            "code_completion": (
                "Complete o seguinte código Python:\n"
                "```python\n{code}\n```"
            ),
            "code_explanation": (
                "Explique o seguinte código Python:\n"
                "```python\n{code}\n```"
            ),
            "code_review": (
                "Revise o seguinte código Python e sugira melhorias:\n"
                "```python\n{code}\n```"
            ),
            "bug_fix": (
                "O seguinte código Python tem um bug. Identifique e corrija:\n"
                "```python\n{code}\n```"
            ),
            "function_generation": (
                "Escreva uma função Python que {description}"
            ),
            "class_generation": (
                "Escreva uma classe Python que {description}"
            ),
            "test_generation": (
                "Escreva testes unitários para o seguinte código Python:\n"
                "```python\n{code}\n```"
            )
        }
    
    def _extract_code_from_response(self, response: str) -> str:
        """
        Extrai blocos de código da resposta.
        
        Args:
            response: Resposta completa
            
        Returns:
            str: Código extraído
        """
        # Procurar por blocos de código com marcação ```python
        code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0].strip()
        
        # Se não encontrar blocos marcados, procurar por indentação consistente
        lines = response.split("\n")
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith("def ") or line.strip().startswith("class "):
                in_code_block = True
                code_lines.append(line)
            elif in_code_block:
                if line.strip() == "" or line.startswith(" ") or line.startswith("\t"):
                    code_lines.append(line)
                else:
                    in_code_block = False
        
        if code_lines:
            return "\n".join(code_lines)
        
        # Se ainda não encontrou código, retornar a resposta completa
        return response
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Gera uma resposta para o prompt.
        
        Args:
            prompt: Prompt para geração
            max_new_tokens: Número máximo de tokens a gerar
            temperature: Temperatura para amostragem
            **kwargs: Argumentos adicionais para geração
            
        Returns:
            str: Resposta gerada
        """
        # Usar configuração padrão se não especificado
        if max_new_tokens is None:
            max_new_tokens = self.code_generation_config["max_new_tokens"]
        
        if temperature is None:
            temperature = self.code_generation_config["temperature"]
        
        # Mesclar configurações
        generation_config = self.code_generation_config.copy()
        generation_config.update(kwargs)
        
        # Tokenizar prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Gerar resposta
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=generation_config["top_p"],
                top_k=generation_config["top_k"],
                repetition_penalty=generation_config["repetition_penalty"],
                do_sample=generation_config["do_sample"],
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decodificar resposta
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remover o prompt da resposta
        response = response[len(prompt):].strip()
        
        # Limpar cache CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response
    
    def complete_code(self, code: str) -> str:
        """
        Completa um trecho de código Python.
        
        Args:
            code: Código a ser completado
            
        Returns:
            str: Código completado
        """
        prompt = self.templates["code_completion"].format(code=code)
        response = self.generate_response(prompt, temperature=0.2)
        return self._extract_code_from_response(response)
    
    def explain_code(self, code: str) -> str:
        """
        Explica um trecho de código Python.
        
        Args:
            code: Código a ser explicado
            
        Returns:
            str: Explicação do código
        """
        prompt = self.templates["code_explanation"].format(code=code)
        return self.generate_response(prompt, temperature=0.7)
    
    def review_code(self, code: str) -> str:
        """
        Revisa um trecho de código Python e sugere melhorias.
        
        Args:
            code: Código a ser revisado
            
        Returns:
            str: Revisão do código
        """
        prompt = self.templates["code_review"].format(code=code)
        return self.generate_response(prompt, temperature=0.7)
    
    def fix_bug(self, code: str) -> str:
        """
        Identifica e corrige bugs em um trecho de código Python.
        
        Args:
            code: Código com bug
            
        Returns:
            str: Código corrigido
        """
        prompt = self.templates["bug_fix"].format(code=code)
        response = self.generate_response(prompt, temperature=0.3)
        return self._extract_code_from_response(response)
    
    def generate_function(self, description: str) -> str:
        """
        Gera uma função Python com base na descrição.
        
        Args:
            description: Descrição da função
            
        Returns:
            str: Função gerada
        """
        prompt = self.templates["function_generation"].format(description=description)
        response = self.generate_response(prompt, temperature=0.3)
        return self._extract_code_from_response(response)
    
    def generate_class(self, description: str) -> str:
        """
        Gera uma classe Python com base na descrição.
        
        Args:
            description: Descrição da classe
            
        Returns:
            str: Classe gerada
        """
        prompt = self.templates["class_generation"].format(description=description)
        response = self.generate_response(prompt, temperature=0.3)
        return self._extract_code_from_response(response)
    
    def generate_tests(self, code: str) -> str:
        """
        Gera testes unitários para um trecho de código Python.
        
        Args:
            code: Código a ser testado
            
        Returns:
            str: Testes unitários
        """
        prompt = self.templates["test_generation"].format(code=code)
        response = self.generate_response(prompt, temperature=0.3)
        return self._extract_code_from_response(response)
    
    def execute_code(self, code: str) -> Tuple[bool, str]:
        """
        Executa um trecho de código Python em um ambiente seguro.
        
        Args:
            code: Código a ser executado
            
        Returns:
            tuple: (sucesso, resultado ou erro)
        """
        import tempfile
        import subprocess
        
        # Criar arquivo temporário
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(code.encode())
            temp_file_path = temp_file.name
        
        try:
            # Executar código com timeout
            result = subprocess.run(
                ["python", temp_file_path],
                capture_output=True,
                text=True,
                timeout=10  # Timeout de 10 segundos
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout: A execução do código excedeu o limite de tempo."
        except Exception as e:
            return False, f"Erro ao executar código: {str(e)}"
        finally:
            # Remover arquivo temporário
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
```

### Interface de Linha de Comando para o Assistente

```python
def run_python_assistant_cli():
    """
    Executa o assistente de programação em modo CLI.
    """
    import argparse
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Assistente de Programação Python")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Nome do modelo a ser usado")
    parser.add_argument("--quantize", type=int, choices=[0, 4, 8], default=8, help="Bits para quantização (0 para desativar)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto", help="Dispositivo para execução")
    parser.add_argument("--mode", type=str, choices=["interactive", "file"], default="interactive", help="Modo de operação")
    parser.add_argument("--file", type=str, help="Arquivo Python para processar (no modo file)")
    parser.add_argument("--task", type=str, choices=["complete", "explain", "review", "fix", "test"], help="Tarefa a ser realizada (no modo file)")
    
    args = parser.parse_args()
    
    # Configurar quantização
    use_8bit = args.quantize == 8
    use_4bit = args.quantize == 4
    
    # Inicializar assistente
    print(f"Inicializando assistente com modelo {args.model}...")
    assistant = PythonCodingAssistant(
        model_name=args.model,
        use_8bit=use_8bit,
        use_4bit=use_4bit,
        device=args.device
    )
    
    # Modo de arquivo
    if args.mode == "file":
        if not args.file or not args.task:
            print("Erro: No modo file, é necessário especificar --file e --task")
            return
        
        # Ler arquivo
        try:
            with open(args.file, "r") as f:
                code = f.read()
        except Exception as e:
            print(f"Erro ao ler arquivo: {str(e)}")
            return
        
        # Executar tarefa
        if args.task == "complete":
            result = assistant.complete_code(code)
            print("\n=== Código Completado ===\n")
        elif args.task == "explain":
            result = assistant.explain_code(code)
            print("\n=== Explicação do Código ===\n")
        elif args.task == "review":
            result = assistant.review_code(code)
            print("\n=== Revisão do Código ===\n")
        elif args.task == "fix":
            result = assistant.fix_bug(code)
            print("\n=== Código Corrigido ===\n")
        elif args.task == "test":
            result = assistant.generate_tests(code)
            print("\n=== Testes Gerados ===\n")
        
        print(result)
    
    # Modo interativo
    else:
        print("Assistente de Programação Python inicializado. Digite 'sair' para encerrar.")
        print("Comandos disponíveis:")
        print("  /completar <código> - Completa um trecho de código")
        print("  /explicar <código> - Explica um trecho de código")
        print("  /revisar <código> - Revisa um trecho de código")
        print("  /corrigir <código> - Corrige bugs em um trecho de código")
        print("  /função <descrição> - Gera uma função com base na descrição")
        print("  /classe <descrição> - Gera uma classe com base na descrição")
        print("  /testar <código> - Gera testes para um trecho de código")
        print("  /executar <código> - Executa um trecho de código")
        print("  /ajuda - Mostra esta mensagem de ajuda")
        
        while True:
            user_input = input("\n> ")
            
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Encerrando assistente...")
                break
            
            if user_input.startswith("/ajuda"):
                print("Comandos disponíveis:")
                print("  /completar <código> - Completa um trecho de código")
                print("  /explicar <código> - Explica um trecho de código")
                print("  /revisar <código> - Revisa um trecho de código")
                print("  /corrigir <código> - Corrige bugs em um trecho de código")
                print("  /função <descrição> - Gera uma função com base na descrição")
                print("  /classe <descrição> - Gera uma classe com base na descrição")
                print("  /testar <código> - Gera testes para um trecho de código")
                print("  /executar <código> - Executa um trecho de código")
                continue
            
            # Processar comandos
            if user_input.startswith("/completar "):
                code = user_input[11:].strip()
                result = assistant.complete_code(code)
                print("\n=== Código Completado ===\n")
                print(result)
            
            elif user_input.startswith("/explicar "):
                code = user_input[10:].strip()
                result = assistant.explain_code(code)
                print("\n=== Explicação ===\n")
                print(result)
            
            elif user_input.startswith("/revisar "):
                code = user_input[9:].strip()
                result = assistant.review_code(code)
                print("\n=== Revisão ===\n")
                print(result)
            
            elif user_input.startswith("/corrigir "):
                code = user_input[10:].strip()
                result = assistant.fix_bug(code)
                print("\n=== Código Corrigido ===\n")
                print(result)
            
            elif user_input.startswith("/função "):
                description = user_input[9:].strip()
                result = assistant.generate_function(description)
                print("\n=== Função Gerada ===\n")
                print(result)
            
            elif user_input.startswith("/classe "):
                description = user_input[8:].strip()
                result = assistant.generate_class(description)
                print("\n=== Classe Gerada ===\n")
                print(result)
            
            elif user_input.startswith("/testar "):
                code = user_input[8:].strip()
                result = assistant.generate_tests(code)
                print("\n=== Testes Gerados ===\n")
                print(result)
            
            elif user_input.startswith("/executar "):
                code = user_input[10:].strip()
                success, result = assistant.execute_code(code)
                if success:
                    print("\n=== Resultado da Execução ===\n")
                else:
                    print("\n=== Erro na Execução ===\n")
                print(result)
            
            else:
                # Tratar como consulta geral
                response = assistant.generate_response(user_input)
                print("\n" + response)
```

## Aplicação Web para Geração de Texto

Vamos criar uma aplicação web simples para geração de texto usando Flask.

### Implementação da Aplicação Web

```python
from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading
import queue
import time
import os

class TextGenerationService:
    """
    Serviço de geração de texto para aplicação web.
    """
    def __init__(
        self,
        model_name: str,
        use_8bit: bool = False,
        use_4bit: bool = False,
        device: str = "auto"
    ):
        """
        Inicializa o serviço de geração de texto.
        
        Args:
            model_name: Nome ou caminho do modelo
            use_8bit: Se True, usa quantização de 8 bits
            use_4bit: Se True, usa quantização de 4 bits
            device: Dispositivo para execução ("cpu", "cuda", "auto")
        """
        # Determinar dispositivo
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Usando dispositivo: {self.device}")
        
        # Carregar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Garantir que o tokenizer tenha tokens especiais
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configurar quantização se solicitado
        if use_4bit or use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                
                if use_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                else:  # use_8bit
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                
                # Carregar modelo quantizado
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device.type == "cuda" else {"": "cpu"}
                )
                
                print(f"Modelo carregado com quantização de {4 if use_4bit else 8} bits")
            except ImportError:
                print("Biblioteca bitsandbytes não disponível, carregando modelo sem quantização")
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.to(self.device)
        else:
            # Carregar modelo sem quantização
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
        
        # Colocar modelo em modo de avaliação
        self.model.eval()
        
        # Fila de requisições
        self.request_queue = queue.Queue()
        self.results = {}
        
        # Iniciar thread de processamento
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_queue(self):
        """
        Processa a fila de requisições em segundo plano.
        """
        while True:
            try:
                # Obter próxima requisição
                request_id, prompt, params = self.request_queue.get()
                
                # Processar requisição
                try:
                    result = self._generate_text(prompt, **params)
                    self.results[request_id] = {"status": "completed", "result": result}
                except Exception as e:
                    self.results[request_id] = {"status": "error", "error": str(e)}
                
                # Marcar requisição como concluída
                self.request_queue.task_done()
                
                # Limpar resultados antigos
                current_time = time.time()
                to_remove = []
                for req_id, result_data in self.results.items():
                    if "timestamp" in result_data and current_time - result_data["timestamp"] > 3600:
                        to_remove.append(req_id)
                
                for req_id in to_remove:
                    del self.results[req_id]
            
            except Exception as e:
                print(f"Erro na thread de processamento: {str(e)}")
                time.sleep(1)  # Evitar loop infinito em caso de erro
    
    def _generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> str:
        """
        Gera texto com base no prompt.
        
        Args:
            prompt: Prompt para geração
            max_length: Comprimento máximo da sequência gerada
            temperature: Temperatura para amostragem
            top_p: Probabilidade cumulativa para nucleus sampling
            top_k: Número de tokens mais prováveis a considerar
            repetition_penalty: Penalidade para repetição de tokens
            do_sample: Se True, amostra da distribuição, caso contrário usa greedy decoding
            num_return_sequences: Número de sequências a retornar
            
        Returns:
            str: Texto gerado
        """
        # Tokenizar prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Gerar texto
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decodificar saída
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Limpar cache CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return generated_texts
    
    def submit_request(self, prompt: str, **params) -> str:
        """
        Submete uma requisição de geração de texto.
        
        Args:
            prompt: Prompt para geração
            **params: Parâmetros adicionais para geração
            
        Returns:
            str: ID da requisição
        """
        # Gerar ID único para a requisição
        request_id = f"req_{int(time.time())}_{len(self.results)}"
        
        # Adicionar requisição à fila
        self.request_queue.put((request_id, prompt, params))
        
        # Registrar requisição nos resultados
        self.results[request_id] = {"status": "pending", "timestamp": time.time()}
        
        return request_id
    
    def get_result(self, request_id: str) -> dict:
        """
        Obtém o resultado de uma requisição.
        
        Args:
            request_id: ID da requisição
            
        Returns:
            dict: Resultado da requisição
        """
        if request_id not in self.results:
            return {"status": "not_found"}
        
        return self.results[request_id]

# Criar aplicação Flask
app = Flask(__name__)

# Inicializar serviço de geração de texto
generation_service = None

@app.route('/')
def index():
    """Página inicial."""
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    """API para geração de texto."""
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt não fornecido"}), 400
    
    # Extrair parâmetros
    prompt = data['prompt']
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    top_k = data.get('top_k', 50)
    repetition_penalty = data.get('repetition_penalty', 1.0)
    do_sample = data.get('do_sample', True)
    
    # Validar parâmetros
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"error": "Prompt inválido"}), 400
    
    if not isinstance(max_length, int) or max_length < 1 or max_length > 1000:
        return jsonify({"error": "max_length deve ser um inteiro entre 1 e 1000"}), 400
    
    # Submeter requisição
    request_id = generation_service.submit_request(
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample
    )
    
    return jsonify({"request_id": request_id})

@app.route('/api/result/<request_id>', methods=['GET'])
def get_result(request_id):
    """API para obter resultado de geração."""
    result = generation_service.get_result(request_id)
    return jsonify(result)

def create_app(model_name="distilgpt2", use_8bit=False, use_4bit=False, device="auto"):
    """
    Cria e configura a aplicação Flask.
    
    Args:
        model_name: Nome ou caminho do modelo
        use_8bit: Se True, usa quantização de 8 bits
        use_4bit: Se True, usa quantização de 4 bits
        device: Dispositivo para execução
        
    Returns:
        Flask: Aplicação Flask configurada
    """
    global generation_service
    
    # Inicializar serviço de geração de texto
    generation_service = TextGenerationService(
        model_name=model_name,
        use_8bit=use_8bit,
        use_4bit=use_4bit,
        device=device
    )
    
    # Criar diretório de templates se não existir
    os.makedirs('templates', exist_ok=True)
    
    # Criar template HTML
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gerador de Texto</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        textarea, input[type="number"], input[type="range"] {
            width: 100%;
            padding: 8px;
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        textarea {
            height: 100px;
            resize: vertical;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            white-space: pre-wrap;
        }
        .loading {
            text-align: center;
            margin-top: 20px;
        }
        .slider-container {
            display: flex;
            align-items: center;
        }
        .slider-container input[type="range"] {
            flex: 1;
            margin-right: 10px;
        }
        .slider-value {
            width: 50px;
            text-align: center;
        }
        .error {
            color: red;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>Gerador de Texto</h1>
    <div class="container">
        <div>
            <label for="prompt">Prompt:</label>
            <textarea id="prompt" placeholder="Digite seu prompt aqui..."></textarea>
        </div>
        
        <div>
            <label for="max-length">Comprimento máximo:</label>
            <input type="number" id="max-length" min="1" max="1000" value="100">
        </div>
        
        <div>
            <label for="temperature">Temperatura: <span id="temperature-value">0.7</span></label>
            <div class="slider-container">
                <input type="range" id="temperature" min="0.1" max="1.5" step="0.1" value="0.7">
            </div>
        </div>
        
        <div>
            <label for="top-p">Top-p (nucleus sampling): <span id="top-p-value">0.9</span></label>
            <div class="slider-container">
                <input type="range" id="top-p" min="0.1" max="1.0" step="0.1" value="0.9">
            </div>
        </div>
        
        <div>
            <label for="top-k">Top-k: <span id="top-k-value">50</span></label>
            <div class="slider-container">
                <input type="range" id="top-k" min="1" max="100" step="1" value="50">
            </div>
        </div>
        
        <button id="generate-btn">Gerar Texto</button>
        
        <div id="loading" class="loading" style="display: none;">
            <p>Gerando texto...</p>
        </div>
        
        <div id="error" class="error" style="display: none;"></div>
        
        <div id="result" class="result" style="display: none;"></div>
    </div>

    <script>
        // Atualizar valores dos sliders
        document.getElementById('temperature').addEventListener('input', function() {
            document.getElementById('temperature-value').textContent = this.value;
        });
        
        document.getElementById('top-p').addEventListener('input', function() {
            document.getElementById('top-p-value').textContent = this.value;
        });
        
        document.getElementById('top-k').addEventListener('input', function() {
            document.getElementById('top-k-value').textContent = this.value;
        });
        
        // Função para gerar texto
        document.getElementById('generate-btn').addEventListener('click', async function() {
            const prompt = document.getElementById('prompt').value.trim();
            
            if (!prompt) {
                showError('Por favor, digite um prompt.');
                return;
            }
            
            const maxLength = parseInt(document.getElementById('max-length').value);
            const temperature = parseFloat(document.getElementById('temperature').value);
            const topP = parseFloat(document.getElementById('top-p').value);
            const topK = parseInt(document.getElementById('top-k').value);
            
            // Mostrar loading e esconder resultado anterior
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('generate-btn').disabled = true;
            
            try {
                // Enviar requisição
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        prompt,
                        max_length: maxLength,
                        temperature,
                        top_p: topP,
                        top_k: topK
                    })
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || 'Erro ao gerar texto');
                }
                
                const requestId = data.request_id;
                
                // Polling para obter resultado
                await pollResult(requestId);
            } catch (error) {
                showError(error.message);
                document.getElementById('loading').style.display = 'none';
                document.getElementById('generate-btn').disabled = false;
            }
        });
        
        // Função para polling do resultado
        async function pollResult(requestId) {
            try {
                const response = await fetch(`/api/result/${requestId}`);
                const data = await response.json();
                
                if (data.status === 'completed') {
                    // Mostrar resultado
                    const resultElement = document.getElementById('result');
                    resultElement.textContent = Array.isArray(data.result) ? data.result[0] : data.result;
                    resultElement.style.display = 'block';
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('generate-btn').disabled = false;
                } else if (data.status === 'error') {
                    showError(data.error || 'Erro ao gerar texto');
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('generate-btn').disabled = false;
                } else if (data.status === 'pending') {
                    // Continuar polling
                    setTimeout(() => pollResult(requestId), 1000);
                } else {
                    showError('Requisição não encontrada');
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('generate-btn').disabled = false;
                }
            } catch (error) {
                showError(error.message);
                document.getElementById('loading').style.display = 'none';
                document.getElementById('generate-btn').disabled = false;
            }
        }
        
        // Função para mostrar erro
        function showError(message) {
            const errorElement = document.getElementById('error');
            errorElement.textContent = message;
            errorElement.style.display = 'block';
        }
    </script>
</body>
</html>
        """)
    
    return app

def run_web_app():
    """
    Executa a aplicação web.
    """
    import argparse
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Aplicação Web para Geração de Texto")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Nome do modelo a ser usado")
    parser.add_argument("--quantize", type=int, choices=[0, 4, 8], default=8, help="Bits para quantização (0 para desativar)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto", help="Dispositivo para execução")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host para servidor")
    parser.add_argument("--port", type=int, default=5000, help="Porta para servidor")
    
    args = parser.parse_args()
    
    # Configurar quantização
    use_8bit = args.quantize == 8
    use_4bit = args.quantize == 4
    
    # Criar aplicação
    app = create_app(
        model_name=args.model,
        use_8bit=use_8bit,
        use_4bit=use_4bit,
        device=args.device
    )
    
    # Executar aplicação
    app.run(host=args.host, port=args.port)
```

## Assistente de Pesquisa com Recuperação de Informações

Vamos criar um assistente de pesquisa que combina LLM com recuperação de informações.

### Implementação do Assistente de Pesquisa

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os
import json
import re
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResearchAssistant:
    """
    Assistente de pesquisa com recuperação de informações.
    """
    def __init__(
        self,
        model_name: str,
        use_8bit: bool = False,
        use_4bit: bool = False,
        device: Optional[str] = None,
        knowledge_dir: str = "./knowledge_base"
    ):
        """
        Inicializa o assistente de pesquisa.
        
        Args:
            model_name: Nome ou caminho do modelo
            use_8bit: Se True, usa quantização de 8 bits
            use_4bit: Se True, usa quantização de 4 bits
            device: Dispositivo para execução ("cpu", "cuda", "auto")
            knowledge_dir: Diretório para base de conhecimento
        """
        # Determinar dispositivo
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Carregar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Garantir que o tokenizer tenha tokens especiais
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configurar quantização se solicitado
        if use_4bit or use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                
                if use_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                else:  # use_8bit
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                
                # Carregar modelo quantizado
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device.type == "cuda" else {"": "cpu"}
                )
                
                print(f"Modelo carregado com quantização de {4 if use_4bit else 8} bits")
            except ImportError:
                print("Biblioteca bitsandbytes não disponível, carregando modelo sem quantização")
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.to(self.device)
        else:
            # Carregar modelo sem quantização
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
        
        # Colocar modelo em modo de avaliação
        self.model.eval()
        
        # Configurar base de conhecimento
        self.knowledge_dir = knowledge_dir
        os.makedirs(knowledge_dir, exist_ok=True)
        
        # Inicializar vetorizador TF-IDF
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Carregar base de conhecimento existente
        self.documents = []
        self.document_vectors = None
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """
        Carrega a base de conhecimento existente.
        """
        index_path = os.path.join(self.knowledge_dir, "index.json")
        
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                
                self.documents = []
                
                for doc_info in index:
                    doc_path = os.path.join(self.knowledge_dir, doc_info["filename"])
                    
                    if os.path.exists(doc_path):
                        with open(doc_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        self.documents.append({
                            "id": doc_info["id"],
                            "title": doc_info["title"],
                            "source": doc_info["source"],
                            "content": content,
                            "date": doc_info.get("date")
                        })
                
                print(f"Carregados {len(self.documents)} documentos da base de conhecimento")
                
                # Vetorizar documentos
                if self.documents:
                    contents = [doc["content"] for doc in self.documents]
                    self.document_vectors = self.vectorizer.fit_transform(contents)
            
            except Exception as e:
                print(f"Erro ao carregar base de conhecimento: {str(e)}")
                self.documents = []
                self.document_vectors = None
    
    def add_document(self, title: str, content: str, source: str, date: Optional[str] = None) -> str:
        """
        Adiciona um documento à base de conhecimento.
        
        Args:
            title: Título do documento
            content: Conteúdo do documento
            source: Fonte do documento
            date: Data do documento (opcional)
            
        Returns:
            str: ID do documento
        """
        # Gerar ID único
        doc_id = f"doc_{len(self.documents)}_{int(time.time())}"
        
        # Criar documento
        document = {
            "id": doc_id,
            "title": title,
            "source": source,
            "content": content,
            "date": date
        }
        
        # Adicionar à lista de documentos
        self.documents.append(document)
        
        # Salvar documento
        filename = f"{doc_id}.txt"
        with open(os.path.join(self.knowledge_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
        
        # Atualizar índice
        index_path = os.path.join(self.knowledge_dir, "index.json")
        
        index = []
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
            except:
                index = []
        
        index.append({
            "id": doc_id,
            "title": title,
            "source": source,
            "filename": filename,
            "date": date
        })
        
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        
        # Atualizar vetores
        if self.document_vectors is None:
            contents = [doc["content"] for doc in self.documents]
            self.document_vectors = self.vectorizer.fit_transform(contents)
        else:
            # Adicionar novo documento aos vetores existentes
            self.document_vectors = self.vectorizer.fit_transform([doc["content"] for doc in self.documents])
        
        return doc_id
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Busca documentos relevantes para a consulta.
        
        Args:
            query: Consulta de busca
            top_k: Número de documentos a retornar
            
        Returns:
            List[Dict]: Lista de documentos relevantes
        """
        if not self.documents or self.document_vectors is None:
            return []
        
        # Vetorizar consulta
        query_vector = self.vectorizer.transform([query])
        
        # Calcular similaridade
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Obter índices dos documentos mais similares
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Filtrar documentos com similaridade muito baixa
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Limiar de similaridade
                doc = self.documents[idx].copy()
                doc["similarity"] = float(similarities[idx])
                results.append(doc)
        
        return results
    
    def fetch_and_add_webpage(self, url: str) -> str:
        """
        Busca uma página web e adiciona à base de conhecimento.
        
        Args:
            url: URL da página
            
        Returns:
            str: ID do documento
        """
        try:
            # Buscar página
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Extrair conteúdo
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remover scripts, estilos e outros elementos não relevantes
            for element in soup(["script", "style", "meta", "noscript", "svg"]):
                element.decompose()
            
            # Extrair título
            title = soup.title.string if soup.title else url
            
            # Extrair texto principal
            paragraphs = soup.find_all("p")
            content = "\n\n".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
            
            # Se não encontrou parágrafos suficientes, usar todo o texto
            if len(content) < 500:
                content = soup.get_text()
                
                # Limpar espaços em branco excessivos
                content = re.sub(r'\s+', ' ', content).strip()
            
            # Adicionar à base de conhecimento
            return self.add_document(
                title=title,
                content=content,
                source=url,
                date=None
            )
        
        except Exception as e:
            print(f"Erro ao buscar página {url}: {str(e)}")
            return None
    
    def generate_response(
        self,
        query: str,
        use_knowledge_base: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> str:
        """
        Gera uma resposta para a consulta.
        
        Args:
            query: Consulta do usuário
            use_knowledge_base: Se True, usa a base de conhecimento
            max_new_tokens: Número máximo de tokens a gerar
            temperature: Temperatura para amostragem
            top_p: Probabilidade cumulativa para nucleus sampling
            top_k: Número de tokens mais prováveis a considerar
            repetition_penalty: Penalidade para repetição de tokens
            do_sample: Se True, amostra da distribuição, caso contrário usa greedy decoding
            
        Returns:
            str: Resposta gerada
        """
        # Buscar documentos relevantes
        relevant_docs = []
        if use_knowledge_base and self.documents:
            relevant_docs = self.search_documents(query, top_k=3)
        
        # Construir prompt
        prompt = "Você é um assistente de pesquisa útil e preciso. "
        
        if relevant_docs:
            prompt += "Responda à pergunta com base nas seguintes informações:\n\n"
            
            for i, doc in enumerate(relevant_docs):
                prompt += f"Documento {i+1} (Fonte: {doc['source']}):\n{doc['content'][:1000]}...\n\n"
            
            prompt += f"Pergunta: {query}\n\nResposta:"
        else:
            prompt += f"Responda à seguinte pergunta da melhor forma possível:\n\nPergunta: {query}\n\nResposta:"
        
        # Tokenizar prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Gerar resposta
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decodificar resposta
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrair apenas a parte gerada
        response = full_response[len(prompt):].strip()
        
        # Limpar cache CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response
    
    def research_topic(self, topic: str, search_urls: List[str] = None) -> Dict[str, Any]:
        """
        Realiza pesquisa sobre um tópico.
        
        Args:
            topic: Tópico a pesquisar
            search_urls: Lista de URLs para buscar (opcional)
            
        Returns:
            Dict: Resultado da pesquisa
        """
        # Adicionar URLs à base de conhecimento
        if search_urls:
            for url in search_urls:
                self.fetch_and_add_webpage(url)
        
        # Gerar perguntas sobre o tópico
        questions_prompt = f"Gere 5 perguntas importantes sobre o tópico: {topic}"
        questions_response = self.generate_response(
            questions_prompt,
            use_knowledge_base=False,
            temperature=0.7
        )
        
        # Extrair perguntas
        questions = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', questions_response, re.DOTALL)
        if not questions:
            questions = questions_response.split('\n')
        
        questions = [q.strip() for q in questions if q.strip()]
        
        # Limitar a 5 perguntas
        questions = questions[:5]
        
        # Responder cada pergunta
        answers = {}
        for question in questions:
            answer = self.generate_response(
                question,
                use_knowledge_base=True,
                temperature=0.3,
                max_new_tokens=512
            )
            answers[question] = answer
        
        # Gerar resumo
        summary_prompt = f"Crie um resumo abrangente sobre {topic} com base nas seguintes informações:\n\n"
        
        for question, answer in answers.items():
            summary_prompt += f"Pergunta: {question}\nResposta: {answer}\n\n"
        
        summary = self.generate_response(
            summary_prompt,
            use_knowledge_base=True,
            temperature=0.3,
            max_new_tokens=1024
        )
        
        # Retornar resultado
        return {
            "topic": topic,
            "questions": questions,
            "answers": answers,
            "summary": summary
        }
```

### Interface de Linha de Comando para o Assistente de Pesquisa

```python
def run_research_assistant_cli():
    """
    Executa o assistente de pesquisa em modo CLI.
    """
    import argparse
    import time
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Assistente de Pesquisa")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Nome do modelo a ser usado")
    parser.add_argument("--quantize", type=int, choices=[0, 4, 8], default=8, help="Bits para quantização (0 para desativar)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto", help="Dispositivo para execução")
    parser.add_argument("--knowledge-dir", type=str, default="./knowledge_base", help="Diretório para base de conhecimento")
    
    args = parser.parse_args()
    
    # Configurar quantização
    use_8bit = args.quantize == 8
    use_4bit = args.quantize == 4
    
    # Inicializar assistente
    print(f"Inicializando assistente com modelo {args.model}...")
    assistant = ResearchAssistant(
        model_name=args.model,
        use_8bit=use_8bit,
        use_4bit=use_4bit,
        device=args.device,
        knowledge_dir=args.knowledge_dir
    )
    
    print("Assistente de Pesquisa inicializado. Digite 'sair' para encerrar.")
    print("Comandos disponíveis:")
    print("  /pesquisar <tópico> - Realiza pesquisa sobre um tópico")
    print("  /adicionar <url> - Adiciona uma página web à base de conhecimento")
    print("  /buscar <consulta> - Busca documentos na base de conhecimento")
    print("  /ajuda - Mostra esta mensagem de ajuda")
    
    while True:
        user_input = input("\n> ")
        
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("Encerrando assistente...")
            break
        
        if user_input.startswith("/ajuda"):
            print("Comandos disponíveis:")
            print("  /pesquisar <tópico> - Realiza pesquisa sobre um tópico")
            print("  /adicionar <url> - Adiciona uma página web à base de conhecimento")
            print("  /buscar <consulta> - Busca documentos na base de conhecimento")
            print("  /ajuda - Mostra esta mensagem de ajuda")
            continue
        
        # Processar comandos
        if user_input.startswith("/pesquisar "):
            topic = user_input[11:].strip()
            
            if not topic:
                print("Por favor, especifique um tópico para pesquisar.")
                continue
            
            print(f"Pesquisando sobre: {topic}")
            print("Isso pode levar alguns minutos...")
            
            start_time = time.time()
            result = assistant.research_topic(topic)
            end_time = time.time()
            
            print(f"\n=== Pesquisa sobre {topic} ===\n")
            print("Perguntas e Respostas:")
            
            for i, question in enumerate(result["questions"]):
                print(f"\nPergunta {i+1}: {question}")
                print(f"Resposta: {result['answers'][question]}")
            
            print("\nResumo:")
            print(result["summary"])
            
            print(f"\n[Tempo de pesquisa: {end_time - start_time:.2f}s]")
        
        elif user_input.startswith("/adicionar "):
            url = user_input[11:].strip()
            
            if not url:
                print("Por favor, especifique uma URL para adicionar.")
                continue
            
            print(f"Adicionando página: {url}")
            
            doc_id = assistant.fetch_and_add_webpage(url)
            
            if doc_id:
                print(f"Página adicionada com sucesso. ID: {doc_id}")
            else:
                print("Erro ao adicionar página.")
        
        elif user_input.startswith("/buscar "):
            query = user_input[8:].strip()
            
            if not query:
                print("Por favor, especifique uma consulta para buscar.")
                continue
            
            print(f"Buscando: {query}")
            
            results = assistant.search_documents(query)
            
            if results:
                print(f"\nEncontrados {len(results)} documentos relevantes:\n")
                
                for i, doc in enumerate(results):
                    print(f"Documento {i+1}: {doc['title']}")
                    print(f"Fonte: {doc['source']}")
                    print(f"Similaridade: {doc['similarity']:.2f}")
                    print(f"Trecho: {doc['content'][:200]}...\n")
            else:
                print("Nenhum documento relevante encontrado.")
        
        else:
            # Tratar como consulta geral
            print("Gerando resposta...")
            
            start_time = time.time()
            response = assistant.generate_response(user_input)
            end_time = time.time()
            
            print("\nResposta:")
            print(response)
            print(f"\n[Tempo de resposta: {end_time - start_time:.2f}s]")
```

## Conclusão

Neste módulo final, aplicamos todo o conhecimento adquirido ao longo do curso para desenvolver projetos práticos completos. Implementamos:

1. Um chatbot com memória de contexto otimizado para hardware limitado
2. Um assistente de programação especializado em Python
3. Uma aplicação web para geração de texto
4. Um assistente de pesquisa com recuperação de informações

Estas aplicações demonstram como é possível criar sistemas úteis e funcionais baseados em LLMs mesmo com recursos computacionais limitados. Utilizamos técnicas de otimização como quantização, offloading e gerenciamento eficiente de memória para permitir a execução desses sistemas em GPUs com apenas 4-6GB de VRAM ou no Google Colab.

Além disso, exploramos como integrar LLMs com outras tecnologias, como recuperação de informações e interfaces web, para criar aplicações mais completas e úteis.

Com o conhecimento adquirido neste curso, você está preparado para desenvolver suas próprias aplicações baseadas em LLMs, adaptadas às suas necessidades específicas e otimizadas para o hardware disponível.

## Exercícios Práticos

1. Estenda o chatbot para suportar múltiplas personalidades que podem ser selecionadas pelo usuário.
2. Implemente um sistema de cache para o assistente de programação que armazene respostas para consultas frequentes.
3. Adicione suporte para upload de arquivos na aplicação web, permitindo que o modelo gere texto com base no conteúdo do arquivo.
4. Estenda o assistente de pesquisa para buscar automaticamente informações na web sobre o tópico consultado.
5. Crie uma interface gráfica para o assistente de programação usando uma biblioteca como PyQt ou Tkinter.
6. Implemente um sistema de avaliação de respostas que permita ao usuário fornecer feedback sobre a qualidade das respostas geradas.
7. Combine os diferentes projetos em uma única aplicação integrada que ofereça múltiplas funcionalidades.
8. Otimize ainda mais o uso de memória implementando um sistema de paginação para modelos muito grandes.
