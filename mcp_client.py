# mcp_client.py

import asyncio
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
import json
import sys


async def main():
    """
    Cliente de ejemplo para interactuar con AgentToolsMCPServer a través de stdio.
    """
    # 1. Usar el cliente MCP para iniciar el servidor como un subproceso
    print("Iniciando servidor MCP como subproceso...")
    # El comando para iniciar el servidor. sys.executable asegura que se use el mismo intérprete de Python.
    transport = PythonStdioTransport("mcp_server.py")

    async with Client(transport) as client:
        print("Conectado al servidor MCP a través de stdio.")
        await client.ping()
        print("Ping exitoso al servidor MCP.")

        # 2. Listar las herramientas disponibles en el servidor
        print("\n--- Herramientas Disponibles ---")
        try:
            tools = await client.list_tools()
            if not tools:
                print("No se encontraron herramientas disponibles en el servidor.")
            else:
                for tool in tools:
                    print(f"- {tool.name}: {tool.description}")
                    # Imprimir argumentos de una manera más limpia si es posible
                    try:
                        args_json = json.dumps(tool.inputSchema, indent=4, ensure_ascii=False)
                        print(f"  Argumentos: {args_json}")
                    except TypeError:
                        print(f"  Argumentos: {tool.inputSchema}")
            print("---------------------------------")

            # 3. Ejemplo de llamada a una herramienta: get_availability
            print("\n--- Llamando a 'get_availability' ---")
            tool_name = "get_availability"
            arguments = {
                "empresa_id": "CLARUSDENT",
                "especialista_id": "DOC001",
                "fecha": "2024-08-15",
                "paciente_id": "PAC007"
            }
            print(f"Llamando a '{tool_name}' con los argumentos: {arguments}")

            result = await client.call_tool(tool_name, arguments)

            print("\nRespuesta del servidor:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("---------------------------------")

            # 4. Ejemplo de llamada a otra herramienta: find_appointments_by_patient
            print("\n--- Llamando a 'find_appointments_by_patient' ---")
            tool_name_2 = "find_appointments_by_patient"
            arguments_2 = {
                "patient_id": "PAC007"
            }
            print(f"Llamando a '{tool_name_2}' con los argumentos: {arguments_2}")

            result_2 = await client.call_tool(tool_name_2, arguments_2)

            print("\nRespuesta del servidor:")
            print(json.dumps(result_2, indent=2, ensure_ascii=False))
            print("---------------------------------")

        except Exception as e:
            print(f"\nOcurrió un error durante la comunicación con el servidor: {e}")
            print("Asegúrate de que 'mcp_server.py' es ejecutable y no tiene errores.")

        finally:
            print("\nCliente finalizado. El servidor subproceso se detendrá automáticamente.")



if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("Cliente finalizado.")
    except KeyboardInterrupt:
        print("\nCliente detenido por el usuario.")
