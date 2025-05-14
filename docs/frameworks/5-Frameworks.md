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

