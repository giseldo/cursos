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

