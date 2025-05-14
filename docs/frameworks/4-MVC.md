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

