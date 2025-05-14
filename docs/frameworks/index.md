---
lang: pt-BR
title: Desenvolvimento de Sistemas com Frameworks
---

# Desenvolvimento de Sistemas com Frameworks

 ## Introdução ao Desenvolvimento com Frameworks

Criado por Alana Neo.

Bem-vindo ao tutorial de \"Desenvolvimento de Sistemas com Frameworks\".
Este curso, com carga horária de 80 horas-aula (60 horas), foi projetado
para fornecer uma compreensão profunda dos conceitos e práticas
envolvidos no desenvolvimento de sistemas utilizando frameworks. Você
aprenderá sobre modelos de mapeamento objeto-relacional (ORM), padrões
de persistência de objetos, padrões de projeto, arquitetura MVC,
frameworks para desenvolvimento desktop e comunicação via REST API.

### Objetivos do Curso

-   Compreender a importância dos frameworks no desenvolvimento de
    sistemas.
-   Utilizar ORM para interagir com bancos de dados de forma eficiente.
-   Aplicar padrões de persistência para gerenciar dados.
-   Implementar padrões de projeto para criar software modular e
    reutilizável.
-   Entender e aplicar a arquitetura MVC em diferentes frameworks.
-   Desenvolver aplicações desktop com frameworks especializados.
-   Criar e consumir serviços web via REST API.

### Estrutura do Curso

O tutorial está dividido em sete seções, cada uma abordando um tópico da
ementa:

1.  Introdução ao Desenvolvimento com Frameworks
2.  Modelos de Mapeamento Objeto-Relacional (ORM)
3.  Padrões de Persistência de Objetos
4.  Padrões de Projeto
5.  Arquitetura MVC
6.  Frameworks para Desenvolvimento Desktop
7.  Comunicação via REST API

Cada seção contém:

-   **Conteúdo Teórico**: Explicações detalhadas dos conceitos.
-   **Exemplos Práticos**: Códigos e cenários reais.
-   **Exercícios**: Atividades para reforçar o aprendizado.
-   **Materiais Didáticos**: Links para recursos adicionais, como
    documentação e tutoriais.

### Material Didático

-   Exemplos de código em linguagens como Java, C#, Python e JavaScript.
-   Exercícios práticos para reforçar o aprendizado.
-   Links para documentação oficial e tutoriais complementares, como
    [Angular](https://angular.io) e [Spring
    Boot](https://spring.io/projects/spring-boot).

### O que são Frameworks?

Um framework é uma estrutura pré-definida que fornece ferramentas,
bibliotecas e convenções para facilitar o desenvolvimento de software.
Ele atua como uma base sobre a qual os desenvolvedores constroem
aplicações, reduzindo a necessidade de criar tudo do zero.

### Vantagens de Usar Frameworks

-   **Produtividade**: Componentes pré-construídos aceleram o
    desenvolvimento.
-   **Manutenção**: Padronização facilita a manutenção do código.
-   **Escalabilidade**: Estruturas organizadas suportam o crescimento da
    aplicação.
-   **Comunidade**: Suporte de comunidades ativas com plugins e
    atualizações.

### Tipos de Frameworks

  Tipo        Descrição                                  Exemplos
  ----------- ------------------------------------------ -------------------------------------------
  Front-end   Desenvolvimento de interfaces de usuário   Angular, React, Vue.js, Streamlit, Gradio
  Back-end    Lógica de negócios e servidor              Spring Boot, Express, Django
  Desktop     Aplicações para computadores               JavaFX, .NET WPF, Qt
  Mobile      Aplicações para dispositivos móveis        React Native, Flutter

### Exercícios

1.  Pesquise três frameworks front-end e compare suas características
    (ex.: facilidade de uso, desempenho).
2.  Compare três frameworks back-end, considerando suporte a bancos de
    dados e escalabilidade.
3.  Identifique três frameworks desktop e analise suas capacidades
    multiplataforma.

### Recursos Adicionais

-   [O que é um
    Framework](https://blog.betrybe.com/framework-de-programacao/o-que-e-framework/)
-   [Frameworks em
    Programação](https://www.escoladnc.com.br/blog/frameworks-em-programacao-conheca-os-principais-e-beneficios/)

## Modelos de Mapeamento Objeto-Relacional (ORM)

O mapeamento objeto-relacional (ORM) é uma técnica que conecta objetos
de uma linguagem de programação a tabelas de bancos de dados
relacionais. Isso permite que desenvolvedores manipulem dados como
objetos, sem escrever consultas SQL complexas.

### Benefícios do ORM

-   **Abstração**: Simplifica interações com o banco de dados.
-   **Portabilidade**: Facilita a troca de bancos de dados.
-   **Segurança**: Reduz riscos de injeção SQL.
-   **Produtividade**: Agiliza operações de banco de dados.

### Exemplos de ORMs

  ORM                Linguagem   Descrição
  ------------------ ----------- ---------------------------------------------
  Hibernate          Java        ORM robusto para aplicações empresariais
  Entity Framework   .NET        Integração nativa com ecossistema Microsoft
  SQLAlchemy         Python      Flexível para diversos bancos de dados
  Doctrine           PHP         Popular em aplicações web PHP

### Exemplo Prático: Hibernate

```python
    @Entity
    @Table(name = "usuarios")
    public class Usuario {
        @Id
        @GeneratedValue(strategy = GenerationType.IDENTITY)
        private Long id;
        private String nome;
        private String email;
        // Getters e Setters
    }
```

### Exercícios

1.  Configure o Hibernate e mapeie uma classe Java para uma tabela de
    banco de dados.
2.  Use o Entity Framework para criar um modelo de dados e realizar
    operações CRUD.
3.  Com o SQLAlchemy, defina um modelo e execute consultas em um banco
    de dados.

### Recursos Adicionais

-   [Primeiros Passos com
    Hibernate](https://hibernate.org/orm/documentation/6.6/)
-   [Entity Framework
    Tutorial](https://learn.microsoft.com/en-us/ef/core/get-started/overview/first-app)

## Padrões de Persistência de Objetos

Padrões de persistência são soluções padronizadas para gerenciar a
persistência de dados, garantindo eficiência e consistência. Eles
abstraem a lógica de acesso a dados, facilitando a manutenção.

### Padrões Comuns

-   **Repository**: Interface para operações CRUD, isolando a lógica de
    acesso ao banco.
-   **Unit of Work**: Gerencia transações, garantindo atomicidade.
-   **Identity Map**: Evita duplicação de objetos, mantendo uma única
    instância por identidade.

### Implementação em Frameworks

-   **Spring Data JPA**: Implementa o padrão Repository com interfaces
    como `JpaRepository`.
-   **Entity Framework**: Usa `DbContext` para gerenciar Unit of Work.

### Exemplo Prático: Spring Data JPA

    @Repository
    public interface UsuarioRepository extends JpaRepository<Usuario, Long> {
        List<Usuario> findByNome(String nome);
    }

### Exercícios

1.  Crie um repositório com Spring Data JPA para gerenciar uma entidade.
2.  Implemente o padrão Unit of Work em .NET com Entity Framework.
3.  Explique como o Identity Map é aplicado em um ORM como Hibernate.

### Recursos Adicionais

-   [Spring Data
    JPA](https://docs.spring.io/spring-data/jpa/docs/current/reference/html/)
-   [Entity Framework Core](https://learn.microsoft.com/en-us/ef/core/)

## Padrões de Projeto

Padrões de projeto são soluções reutilizáveis para problemas comuns no
desenvolvimento de software. Eles promovem modularidade, flexibilidade e
manutenibilidade.

### Tipos de Padrões

  Categoria         Exemplos                    Descrição
  ----------------- --------------------------- ------------------------------------
  Criacionais       Singleton, Factory Method   Gerenciam a criação de objetos
  Estruturais       Adapter, Composite          Organizam a composição de classes
  Comportamentais   Observer, Strategy          Gerenciam interações entre objetos

### Exemplos e Implementações

**Singleton**:
```python
    public class DatabaseConnection {
        private static DatabaseConnection instance;
        private DatabaseConnection() {}
        public static DatabaseConnection getInstance() {
            if (instance == null) {
                instance = new DatabaseConnection();
            }
            return instance;
        }
    }
```

**Factory Method**: Define uma interface para criar objetos, permitindo
subclasses decidirem o tipo.

**Observer**: Notifica dependentes sobre mudanças de estado.

### Exercícios

1.  Implemente o padrão Singleton em Java para uma conexão de banco de
    dados.
2.  Crie um Factory Method em C# para instanciar diferentes tipos de
    veículos.
3.  Desenvolva o padrão Observer em Python para notificar eventos.

### Recursos Adicionais

-   [Padrões de Projeto](https://refactoring.guru/design-patterns)
-   [Design Patterns](https://sourcemaking.com/design_patterns)

## Arquitetura MVC

A arquitetura MVC (Model-View-Controller) separa uma aplicação em três
componentes:

-   **Model**: Gerencia dados e lógica de negócios.
-   **View**: Exibe a interface do usuário.
-   **Controller**: Processa entradas do usuário e coordena Model e
    View.

### Implementação em Frameworks

-   **Angular**: Componentes (View), serviços (Model) e controladores
    (Controller).
-   **Spring MVC**: Beans (Model), JSP/Thymeleaf (View) e classes
    `@Controller`.

### Exemplo Prático: Angular

```python
    @Component({
        selector: 'app-usuario',
        template: '<h1>{{ usuario.nome }}</h1>'
    })
    export class UsuarioComponent {
        usuario = { nome: 'João' };
    }
```

### Exercícios

1.  Crie uma aplicação Angular com um componente, serviço e controlador.
2.  Desenvolva uma aplicação Spring MVC para listar e adicionar
    usuários.

### Recursos Adicionais

-   [Guia Completo de Angular](https://angular.io/guide/architecture)
-   [Spring MVC
    Tutorial](https://spring.io/guides/gs/serving-web-content/)

## Frameworks para Desenvolvimento Desktop

Frameworks desktop permitem criar aplicações que rodam diretamente no
computador, com interfaces ricas e integração ao sistema operacional.

### Exemplos

  Framework   Linguagem   Descrição
  ----------- ----------- ---------------------------------------------
  JavaFX      Java        Sucessor do Swing, para interfaces modernas
  .NET WPF    .NET        Interfaces ricas para Windows
  Qt          C++         Aplicações multiplataforma

### Exemplo Prático: JavaFX
```python
    public class HelloWorld extends Application {
        @Override
        public void start(Stage stage) {
            Button btn = new Button("Clique!");
            StackPane root = new StackPane(btn);
            Scene scene = new Scene(root, 300, 250);
            stage.setScene(scene);
            stage.show();
        }
    }
```

### Exercícios

1.  Crie uma aplicação JavaFX com uma janela e um botão.
2.  Desenvolva uma aplicação .NET WPF para exibir uma lista de itens.
3.  Implemente uma aplicação Qt com uma mensagem de boas-vindas.

### Recursos Adicionais

-   [JavaFX Tutorial](https://openjfx.io/)
-   [WPF
    Tutorial](https://learn.microsoft.com/en-us/dotnet/desktop/wpf/get-started/create-wpf-app-visual-studio)

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
