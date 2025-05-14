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

