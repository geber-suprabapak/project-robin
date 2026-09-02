# Codebase Map

Project Robin is the private face-verification API used by Astra. It validates image requests, uses ONNX/OpenCV inference, and stores/searches per-user embeddings in Qdrant.

| Area | Read |
| --- | --- |
| Runtime/API boundary | [Architecture overview](architecture/overview.md) |
| Routes, core, services | [Modules](modules.md) |
| Identification | [Face verification](flows/face-verification.md) |
| Rules | [Invariants](invariants.md) |
