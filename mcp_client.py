# mcp_client.py

import asyncio
from fastmcp import MCPClient
import json

# Define la dirección del servidor MCP.
# Asegúrate de que coincida con la configuración de tu mcp_server.py.
SERVER_HOST = "localhost"
SERVER_PORT = 8765

async def main():
    """
    Cliente de ejemplo para interactuar con AgentToolsMCPServer.
    """
    # 1. Crear una instancia del cliente MCP
    client = MCPClient(server_host=SERVER_HOST, server_port=SERVER_PORT)

    try:
        # 2. Conectar al servidor
        await client.connect()
        print(f"Conectado al servidor MCP en {SERVER_HOST}:{SERVER_PORT}")

        # 3. Listar las herramientas disponibles en el servidor
        print("\n--- Herramientas Disponibles ---")
        tools = await client.list_tools()
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
            print(f"  Argumentos: {json.dumps(tool.arguments, indent=4)}")
        print("---------------------------------")

        # 4. Ejemplo de llamada a una herramienta: get_availability
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
        if result.stderr:
            print(f"Error: {result.stderr}")
        else:
            # El stdout es un string JSON, lo parseamos para una mejor visualización
            try:
                data = json.loads(result.stdout)
                print(json.dumps(data, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(result.stdout)
        print("---------------------------------")

        # 5. Ejemplo de llamada a otra herramienta: find_appointments_by_patient
        print("\n--- Llamando a 'find_appointments_by_patient' ---")
        tool_name_2 = "find_appointments_by_patient"
        arguments_2 = {
            "patient_id": "PAC007"
        }
        print(f"Llamando a '{tool_name_2}' con los argumentos: {arguments_2}")

        result_2 = await client.call_tool(tool_name_2, arguments_2)

        print("\nRespuesta del servidor:")
        if result_2.stderr:
            print(f"Error: {result_2.stderr}")
        else:
            try:
                data_2 = json.loads(result_2.stdout)
                print(json.dumps(data_2, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(result_2.stdout)
        print("---------------------------------")

    except ConnectionRefusedError:
        print(f"Error: No se pudo conectar al servidor en {SERVER_HOST}:{SERVER_PORT}.")
        print("Asegúrate de que 'mcp_server.py' se esté ejecutando.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
    finally:
        # 6. Desconectar del servidor
        if client.is_connected():
            await client.cleanup()
            print("\nDesconectado del servidor MCP.")

if __name__ == "__main__":
    # Ejecutar el cliente
    # Nota: Si estás en un entorno como Jupyter, puede que necesites
    # configurar el bucle de eventos de asyncio de forma diferente.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCliente detenido por el usuario.")
