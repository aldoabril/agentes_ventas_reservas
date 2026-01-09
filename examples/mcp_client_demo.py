# mcp_client.py

import asyncio
from fastmcp import Client
import json
import sys




async def main():
    """
    Cliente de ejemplo para interactuar con AgentToolsMCPServer a través de HTTP.
    """
    # 1. Configurar el cliente para conectarse al servidor MCP
    print("Conectando al servidor MCP en http://localhost:8000/mcp...")
    
    async with Client("http://localhost:8000/mcp") as client:
        # 2. Listar las herramientas disponibles en el servidor
        print("\n--- Herramientas Disponibles ---")
        try:
            tools = await client.list_tools()
            if not tools:
                print("No se encontraron herramientas disponibles en el servidor.")
            else:
                for tool in tools:
                    #print(f"- {tool.name}: {tool.description}")
                    # Imprimir argumentos de una manera más limpia si es posible
                    try:
                        args_json = json.dumps(tool.inputSchema, indent=4, ensure_ascii=False)
                        #print(f"  Argumentos: {args_json}")
                    except TypeError:
                        print(f"  Argumentos: {tool.inputSchema}")
            #print("---------------------------------")

            # 3. Ejemplo de llamada a una herramienta: get_availability
            #print("\n--- Llamando a 'get_availability' ---")
            tool_name = "get_availability"
            arguments = {
                "empresa_id": "A0OZsgiMQQVtwxMhN6Um",
                "especialista_id": "GMcKghlgHvTkoPxj9t4X",
                "fecha": "2025-11-18"
            }
            print(f"Llamando a '{tool_name}' con los argumentos: {arguments}")

            result = await client.call_tool(tool_name, arguments)

            print("\nRespuesta del servidor:")
            # MCP CallToolResult has fields: content (list), structuredContent (optional), isError
            # Prefer structuredContent when available, otherwise fall back to content.
            try:
                if getattr(result, "structuredContent", None) is not None:
                    print(json.dumps(result.structuredContent, indent=2, ensure_ascii=False))
                else:
                    # content may be a list of ContentBlock models or plain dicts
                    content = getattr(result, "content", None)
                    if content is None:
                        print(json.dumps({"isError": getattr(result, "isError", False)}, ensure_ascii=False))
                    else:
                        # Convert any pydantic models to dicts for JSON serialization
                        serializable = []
                        for c in content:
                            try:
                                serializable.append(c.model_dump() if hasattr(c, "model_dump") else (c.dict() if hasattr(c, "dict") else c))
                            except Exception:
                                serializable.append(c)
                        print(json.dumps(serializable, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"Error al procesar la respuesta del servidor: {e}")
            print("---------------------------------")
            print("---------------------------------")

            

        except Exception as e:
            print(f"\nOcurrió un error durante la comunicación con el servidor: {e}")
            print("Asegúrate de que 'mcp_server.py' está corriendo en http://localhost:8000")



if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("Cliente finalizado.")
    except KeyboardInterrupt:
        print("\nCliente detenido por el usuario.")
