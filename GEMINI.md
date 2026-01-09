# Agente de IA para consultorios dentales

Es un agente IA que atiende a los clientes por mensajería

## Tecnologias utilizadas

- Langchain
- Python
- Langgraph
- ChromeDB
- MCP Tools

## Arquitectura del sistema

- Inicialmente se ejecuta en langgraph en forma local
- Se conecta a OPENAI u otros LLM a través de API
- Patrones SOLID
- Buenas practicas de desarrollo en Python

## Reglas de Negocio

1. Se identifica si el usuario es nuevo o existente
2. Si es nuevo se brinda información importante con el fin de converser al cliente para que agende una cita.
3. Si es cliente existente se le ofrece promociones y/o se le recuerda al cliente de sus sesiones

## Paginas de referencia

- https://python.langchain.com/docs/tutorials/chatbot/
- https://python.langchain.com/docs/tutorials/rag/
- https://python.langchain.com/docs/tutorials/qa_chat_history/
- https://python.langchain.com/docs/tutorials/agents/

# Proyecto: Agente Autónomo de Ventas y Reservas

## Dominio

### Descripción:

- **Communication Layer**: Conversacional
- **Idioma**: Español
- **Entorno**: WhatsApp, Messenger y Facebook

### Domain:

Profesionales médicos y odontólogos interesados en adquirir un agente de gestión de citas.

### Context:

Venta consultiva y cierre de reservas. El agente interactúa con prospectos (aún no pagan) o clientes actuales.

### Objective:

Permitir que los clientes interesados resuelvan sus dudas y gestionen reservas de citas de manera totalmente autónoma, sin errores ni conflictos de agenda, garantizando una experiencia fluida, precisa y confiable desde la primera interacción hasta la confirmación final.

El sistema busca optimizar el proceso de atención y agendamiento, reduciendo la intervención humana, asegurando coherencia entre disponibilidad real y reservas confirmadas, y mejorando la satisfacción del cliente mediante respuestas claras y consistentes.

## Arquitectura de Agentes (Multi-Agent System)

El sistema está compuesto por un Lead Qualifier (guardrail inicial), un Orchestrator, 3 agentes especializados y 1 agente guardián (Judge).

El Lead Qualifier es el primer punto de contacto y se encarga de filtrar y validar los mensajes entrantes antes de derivarlos al Orchestrator, que coordina el flujo conversacional y decide qué agente debe intervenir en cada caso.

### 1) Lead Qualifier & Guardrail

**Rol:**
Primer punto de contacto con el cliente. Evalúa la intención del mensaje (consulta, reserva, reprogramación, cancelación) y filtra posibles mensajes no válidos, spam o actores maliciosos.

**Funciones principales:**

- Clasificar la intención del mensaje.
- Detectar y filtrar mensajes no deseados.
- Enriquecer el contexto del lead y registrarlo en el CRM.
- Derivar conversaciones válidas al Orchestrator.

**Input:** Mensaje del cliente.  
**Output:** `{intención, datos mínimos, prioridad, señales de riesgo}` → Orchestrator.

### 2) Orchestrator (Router + Agente de Coordinación)

**Rol:**
Coordina el flujo entre los demás agentes. Recibe el input del Lead Qualifier y decide cuál agente debe intervenir a continuación.

**Funciones principales:**

- Enrutar la conversación según intención y contexto.
- Mantener el estado conversacional global.
- Resolver ambigüedades y conflictos entre agentes.
- Escalar al Guardian o a un humano en caso de riesgo.

**Input:** `{intención, contexto, estado conversacional actual}`.  
**Output:** `{siguiente agente, acción sugerida, motivo de la decisión}`.

### 3) Knowledge Concierge (RAG-agnóstico al dominio)

**Rol:**
Atiende dudas frecuentes y solicitudes de información, utilizando una base de conocimiento genérica que incluye precios, productos o servicios, políticas, cobertura, ubicación, tiempos y Términos y Condiciones.

**Input:** `{pregunta, contexto del cliente}`.  
**Output:** `{respuesta verificada, fuente de referencia, nivel de confianza}` → Orchestrator.

### 4) Negotiator & Objection Handler

**Rol:**
Gestiona objeciones relacionadas con precio, disponibilidad o condiciones. Aplica reglas de negocio predefinidas, propone alternativas dentro de los límites permitidos o deriva al Guardian o a un asesor humano.

**Input:** `{objeción detectada, políticas vigentes}`.  
**Output:** `{propuesta válida y trazable, o solicitud de derivación}` → Orchestrator.

### 5) Scheduler & Conflict Resolver

**Rol:**
Se encarga de la gestión de reservas, reprogramaciones y cancelaciones. Consulta la disponibilidad en tiempo real mediante la API de calendario y ejecuta verificaciones automáticas para evitar conflictos de agenda.

**Controles clave:**

- Chequeo de concurrencia (locks optimistas/pesimistas o tokens de idempotencia).
- Reintentos automáticos con backoff si la disponibilidad cambia durante el proceso.
- Validación de huso horario (TZ) y capacidad máxima del sistema.

**Input:** `{preferencias del cliente, ventanas disponibles}`.  
**Output:** `{slot propuesto/confirmado, ID de reserva, instrucciones finales}` → Orchestrator.

### 6) Guardian Agent (Judge & Compliance)

**Rol:**
Opera como la capa de seguridad y cumplimiento del sistema. Supervisa toda acción crítica antes de su ejecución.

**Validaciones:**

- Consistencia entre el mensaje y las políticas/TyC.
- Coincidencia exacta entre el slot confirmado y la disponibilidad actual.
- Tono y coherencia del mensaje final.

**Input:** `{acción propuesta + evidencias (logs, fuente RAG, respuesta del Scheduler)}`.  
**Output:** `{aprobado | rechazado | escalar a humano}` + justificación.

## Knowledge Base

- **Datos del producto**: Precios, disponibilidad, versiones, promociones, políticas de cancelación.
- **Datos del cliente**: Historial de interacciones, preferencias, nivel de interés, etapa del embudo.
- **Fuente de información**: RAG (Retrieval-Augmented Generation) conectado al CRM y documentos internos.
- **Actualización diaria** con cambios en precios u ofertas.

## Tools e Integraciones

- **API de WhatsApp y Messenger**: Comunicación bidireccional con leads.
- **API de Calendario / Agenda**: Consulta de disponibilidad y reserva automática.
- **API de Pasarela de Pago**: Procesamiento de pagos o confirmaciones de demo.
- **CRM**: Registro de leads, scoring de interés, y seguimiento del embudo.

## Memory

- **Short Term Memory**: Últimas 5 interacciones del cliente (preguntas recientes, objeciones, deseos de reserva).
- **Long Term Memory**: Historial completo de conversaciones, compras previas, datos demográficos y patrones de comportamiento.
- **Persistencia**: Base de datos + embeddings vectoriales para recuperación contextual.

### Convenciones de implementación (proyecto)

- Short-term memory (conversacional) se guarda por `thread_id` usando `langgraph.checkpoint.memory.MemorySaver`.

  - El proyecto usa una única instancia de `MemorySaver` pasada a `workflow.compile(checkpointer=memory)`.
  - La convención es mantener una ventana corta (5 mensajes) en `AgentState["messages"]` para evitar contextos demasiado grandes.
  - Ejemplo de uso en `main.py`: `config = {"configurable": {"thread_id": thread_id}}` y la aplicación gestiona guardado/recupero.

- Long-term memory (persistente): usar el vector store `Chroma` (`knowledge/retriever.py`) indexando embeddings con `user_id`/`thread_id` en metadata.

  - Recuperar contexto relevante via el `retriever` en el `Knowledge Concierge` para respuestas RAG.

- Agent-local ephemeral state: almacenar dentro de `AgentState` bajo claves namespaced (ej.: `negotiator.history`, `scheduler.lock_token`).
  - Evitar crear múltiples `MemorySaver` por agente; usar una memoria compartida y namespaces para aislar datos.

Estas convenciones están reflejadas en `examples/legacy_main.py`, `workflows/conversation_graph.py` y en la nueva utilería `agents/memory.py`.

## Dimensión de Autonomía

- **Tipo**: Semi-Autónomo con Supervisión Controlada.
- Los agentes colaboran entre sí de forma autónoma (sin intervención humana directa).
- El Guardian Agent valida acciones críticas y puede derivar a un humano si detecta riesgo.

## Dimensión de Criticalidad

- **Tipo**: Media.
- Un error puede implicar pérdida de venta o mala experiencia del cliente, pero no riesgo financiero grave.
- Se prioriza la seguridad reputacional y coherencia del mensaje.

## Análisis de Dimensiones: ¿Cómo va a ser su actuar?

**Persuadiendo y guiando hacia una interacción eficiente y confiable:**

1. **Filtra**: El Lead Qualifier analiza el mensaje, filtra spam y valida que el lead cumpla con los criterios mínimos para continuar.
2. **Coordina**: El Orchestrator recibe la información validada y determina qué agente debe intervenir según la intención y el estado de la conversación.
3. **Informa**: El Knowledge Concierge responde dudas y proporciona información verificada desde la base de conocimiento.
4. **Negocia**: El Negotiator gestiona objeciones o solicitudes especiales, aplicando las políticas vigentes.
5. **Cierra**: El Scheduler agenda, reprograma o confirma la reserva, asegurando que no existan conflictos de horario.
6. **Valida**: El Guardian Agent revisa la coherencia, cumplimiento de políticas y tono final del mensaje.

## Posibles Riesgos

- El agente ofrece precios erróneos o beneficios inexistentes.
- Promete horarios no disponibles o genera doble reserva.
- Usa un tono inadecuado (molestia, malentendido).

## Mecanismos de Control

- **Lead Qualifier**: filtro inicial de spam y validación de intención.
- **Guardian Agent**: validación final antes de ejecutar acciones críticas.
- **Reglas de fallback**:
  - Si tras 3 turnos no se entiende la intención → derivar a asesor humano.
  - Si el cliente expresa frustración o solicita hablar con alguien → transferencia inmediata.
- **Registro y trazabilidad**: todas las interacciones se almacenan para auditoría y mejora continua.

## Métricas de Éxito (KPIs)

- **Tasa de reservas confirmadas** = reservas concretadas / conversaciones iniciadas.
- **Tasa de error detectado por el Guardian Agent**.
- **Nivel de satisfacción del cliente** (post-chat rating).
- **Tiempo promedio por reserva**.
