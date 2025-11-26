# Agente de IA para Consultorios Dentales

Sistema multi-agente de IA para gestión de ventas y reservas de citas en consultorios dentales.

## 🏗️ Arquitectura del Proyecto

```
agentes_ventas_reservas/
│
├── config/                      # Configuración centralizada
│   ├── __init__.py
│   └── settings.py              # Variables globales (EMPRESA_ID, API_BASE_URL, etc.)
│
├── agents/                      # Agentes especializados
│   ├── __init__.py
│   ├── base.py                  # Estados y tipos base (AgentState, Intent, RouteQuery)
│   ├── orchestrator.py          # Router que clasifica intención y enruta
│   ├── knowledge_concierge.py   # Responde consultas usando RAG
│   └── scheduler.py             # Gestiona agendamiento de citas
│
├── tools/                       # Herramientas para los agentes
│   ├── __init__.py
│   └── appointment_tools.py     # Herramientas de agendamiento (MCP integration)
│
├── integrations/                # Integraciones con servicios externos
│   ├── __init__.py
│   └── calendar_api.py          # Cliente de API de citas médicas
│
├── knowledge/                   # Base de conocimiento y RAG
│   ├── __init__.py
│   ├── retriever.py             # Sistema RAG para consultas
│   ├── rag_loader.py            # Carga de documentos PDF
│   └── documents/               # Documentos fuente (PDFs)
│
├── workflows/                   # Flujos de trabajo (LangGraph)
│   ├── __init__.py
│   └── conversation_graph.py    # Definición del grafo de conversación
│
├── data/                        # Datos persistentes
│   └── chroma_langchain_db/     # Base de datos vectorial
│
├── examples/                    # Ejemplos y código legacy
│   ├── simple_chatbot.py        # Chatbot simple de ejemplo
│   ├── mcp_client_demo.py       # Cliente MCP de demostración
│   ├── legacy_agents.py         # Versión anterior de agents.py
│   ├── legacy_main.py           # Versión anterior de main.py
│   └── legacy_tools.py          # Herramientas antiguas
│
├── tests/                       # Tests (por implementar)
│   └── __init__.py
│
├── main.py                      # Aplicación principal
├── mcp_server.py                # Servidor MCP (standalone)
├── .env                         # Variables de entorno
└── pyproject.toml               # Dependencias del proyecto
```

## 🚀 Inicio Rápido

### 1. Configurar Variables de Entorno

Copia `.env.sample` a `.env` y configura tus API keys:

```bash
OPENAI_API_KEY=tu_api_key_aqui
LANGSMITH_API_KEY=tu_langsmith_key_aqui
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
# o si usas poetry:
poetry install
```

### 3. Ejecutar la Aplicación

```bash
python main.py
```

### 4. Ejecutar el Servidor MCP (opcional)

En otra terminal:

```bash
python mcp_server.py
```

## 🤖 Sistema Multi-Agente

### Agentes Principales

1. **Orchestrator (Router)**
   - Clasifica la intención del usuario
   - Enruta al agente apropiado
   - Ubicación: `agents/orchestrator.py`

2. **Knowledge Concierge**
   - Responde consultas generales usando RAG
   - Accede a documentos PDF en `knowledge/documents/`
   - Ubicación: `agents/knowledge_concierge.py`

3. **Scheduler**
   - Gestiona agendamiento de citas
   - Verifica disponibilidad
   - Crea y confirma citas
   - Ubicación: `agents/scheduler.py`

### Flujo de Conversación

```
Usuario → Orchestrator → [Knowledge Concierge | Scheduler] → Respuesta
```

## 🛠️ Tecnologías

- **LangChain**: Framework para aplicaciones con LLMs
- **LangGraph**: Orquestación de agentes multi-paso
- **ChromaDB**: Base de datos vectorial para RAG
- **OpenAI**: Modelos de lenguaje (GPT-4o-mini, GPT-3.5-turbo)
- **FastMCP**: Servidor MCP para herramientas
- **Python 3.10+**

## 📝 Configuración

Toda la configuración centralizada está en `config/settings.py`:

- `EMPRESA_ID`: ID de la empresa
- `PACIENTE_ID`: ID del paciente
- `API_BASE_URL`: URL de la API de citas
- `MCP_SERVER_URL`: URL del servidor MCP
- `ESPECIALISTAS`: Diccionario de especialistas disponibles

## 🧪 Testing

```bash
# Verificar imports
python -c "from agents import router_node, knowledge_concierge_node, scheduler_node; print('✓ OK')"

# Ejecutar tests (cuando estén implementados)
pytest tests/
```

## 📚 Documentación Adicional

- **Arquitectura del Sistema**: Ver `GEMINI.md` para detalles completos
- **Plan de Implementación**: Ver `.gemini/antigravity/brain/*/implementation_plan.md`

## 🔧 Desarrollo

### Agregar un Nuevo Agente

1. Crear archivo en `agents/nuevo_agente.py`
2. Definir función `nuevo_agente_node(state: AgentState) -> AgentState`
3. Exportar en `agents/__init__.py`
4. Agregar al grafo en `workflows/conversation_graph.py`

### Agregar Nuevas Herramientas

1. Crear función en `tools/` con decorador `@tool`
2. Agregar a la lista de herramientas del agente correspondiente
3. Documentar con docstring descriptivo

## 📄 Licencia

[Tu licencia aquí]

## 👥 Autores

[Tu nombre aquí]
