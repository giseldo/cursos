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

