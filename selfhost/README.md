# Self-host bootstrap

Os primeiros componentes do compilador Nano agora estão escritos em Nano:

- selfhost/lexer.nano: leitura de caracteres, comentários, identificadores, números, strings e símbolos.
- selfhost/parser.nano: AST para expressões, chamadas, listas, objetos, funções e controle de fluxo.
- selfhost/compiler.nano: AST para IR textual com labels simbólicos.

Uso conceitual:

1. execute o lexer em Nano para produzir tokens;
2. execute o parser para produzir a AST;
3. execute o compiler para produzir IR textual;
4. compare esse IR com o IR produzido pelo bootstrap Rust.

Este é o começo do bootstrap. O compilador Rust continua sendo a referência até que os dois pipelines passem pelos mesmos testes de conformidade.