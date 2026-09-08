# Trabalho futuro: adaptação dos contatos do DreamerRL

Status: bloqueador aberto da transferência física.

O checkpoint espera quatro booleanos na ordem das pastilhas. No simulador, cada
valor indica força diferente de zero entre uma pastilha específica e um objeto
interativo; contato com mesa ou robô é filtrado. Esforço externo de junta no
driver Trossen não tem automaticamente esse significado.

Antes da inferência com atuação, a solução deve:

- definir sensor ou estimador e mapear exatamente quatro pastilhas;
- distinguir, ou quantificar a incapacidade de distinguir, bola, mesa e robô;
- medir falso positivo, falso negativo e latência em contato e ausência;
- versionar limiar, calibração e transformação;
- registrar todo dado real ou supervisão usado na adaptação;
- validar em posições de desenvolvimento e congelar antes do teste;
- fornecer os quatro canais, sua origem e a versão a cada observação.

`DreamerAdapter` e `DreamerPolicySource` recusam observações sem esses campos.
Não há fallback para zeros, esforço bruto ou estimativa sem versão. A adaptação
não modifica arquitetura, pesos da curiosidade, recompensa, cena ou critério de
sucesso do DreamerRL.
