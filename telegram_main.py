"""
Punto de entrada para el bot de Telegram.
Ejecuta el bot y maneja los mensajes de usuarios.
"""

import asyncio
import logging
from dotenv import load_dotenv

from telegram_bot import create_bot_application
from utils.logger import setup_logging, get_logger
from config import TELEGRAM_BOT_TOKEN

# Cargar variables de entorno
load_dotenv()

# Configurar logging
setup_logging()
logger = get_logger(__name__)


def main():
    """
    Función principal que inicia el bot de Telegram.
    """
    try:
        # Verificar que el token esté configurado
        if not TELEGRAM_BOT_TOKEN:
            logger.error(
                "TELEGRAM_BOT_TOKEN no está configurado. "
                "Por favor configúralo en tu archivo .env"
            )
            return
        
        # Crear la aplicación del bot
        logger.info("Iniciando bot de Telegram...")
        application = create_bot_application()
        
        # Iniciar el bot con polling
        logger.info("Bot iniciado. Esperando mensajes...")
        application.run_polling(
            drop_pending_updates=True
        )
        
    except KeyboardInterrupt:
        logger.info("Bot detenido por el usuario")
    except Exception as e:
        logger.error(f"Error ejecutando el bot: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

