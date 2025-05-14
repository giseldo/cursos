## Comunicação via REST API

REST (Representational State Transfer) é um estilo arquitetural que usa
HTTP para comunicação entre sistemas, manipulando recursos via URIs.

### Princípios do REST

-   **Stateless**: Requisições são independentes.
-   **Client-Server**: Separação entre cliente e servidor.
-   **Cache**: Respostas podem ser armazenadas para desempenho.
-   **Uniform Interface**: Operações padrão (GET, POST, PUT, DELETE).

### Criando e Consumindo REST APIs

-   **Criação**: Frameworks como Spring Boot e Express.
-   **Consumo**: Bibliotecas como Axios e RestTemplate.

### Exemplo Prático: Spring Boot REST API
```python
    @RestController
    @RequestMapping("/api/usuarios")
    public class UsuarioController {
        @GetMapping
        public List<Usuario> listar() {
            return Arrays.asList(new Usuario("João"), new Usuario("Maria"));
        }
    }
```
### Exercícios

1.  Crie uma REST API com Spring Boot para operações CRUD.
2.  Consuma a API do exercício anterior com Axios em JavaScript.
3.  Adicione autenticação JWT à sua REST API.

### Recursos Adicionais

-   [Spring Boot REST API](https://spring.io/guides/gs/rest-service/)
-   [Axios Tutorial](https://axios-http.com/docs/intro)

## Conclusão

Este tutorial fornece uma base sólida para o desenvolvimento de sistemas
com frameworks. Com os conhecimentos adquiridos, você está preparado
para criar aplicações robustas, aplicar padrões de projeto e integrar
sistemas via REST API. Continue explorando os recursos fornecidos para
aprofundar seu aprendizado.


© 2025 Alana Neo. Desenvolvimento de Sistemas com Frameworks
