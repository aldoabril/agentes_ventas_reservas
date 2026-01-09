"""
Bot de Telegram para el sistema de agentes de ventas y reservas.
Maneja mensajes de usuarios de Telegram y los procesa a través del workflow LangGraph.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from langchain_core.messages import HumanMessage
from workflows.conversation_graph import create_app
from agents.memory import truncate_messages
from config import TELEGRAM_BOT_TOKEN
from utils.logger import setup_logging, get_logger

# Configurar logging
setup_logging()
logger = get_logger(__name__)

# Crear la aplicación del grafo (se crea una vez y se reutiliza)
app = create_app()

# Executor para ejecutar el workflow síncrono desde código asíncrono
executor = ThreadPoolExecutor(max_workers=5)


def process_user_message(user_message: str, thread_id: str) -> str:
    """
    Procesa un mensaje del usuario a través del workflow LangGraph.
    
    Args:
        user_message: El mensaje del usuario
        thread_id: ID único del thread para mantener la conversación
        
    Returns:
        La respuesta del agente
    """
    logger.info(f"Procesando mensaje para thread_id: {thread_id}")
    
    # Configuración del thread para mantener la memoria conversacional
    config = {"configurable": {"thread_id": thread_id}}
    
    # Crear input con el mensaje del usuario
    inputs = {"messages": [HumanMessage(content=user_message)]}
    
    # Ejecutar el workflow y obtener la respuesta final
    final_response = None
    last_ai_message = None
    
    try:
        for event in app.stream(inputs, config, stream_mode="values"):
            # Truncar memoria si es necesario
            try:
                event_messages = event.get("messages", [])
                if not event_messages:
                    continue
                    
                # Truncar memoria si es necesario
                truncated = truncate_messages(event_messages)
                if truncated is not None:
                    event["messages"] = truncated
                    
                # Obtener el último mensaje que no sea del usuario
                for msg in reversed(event["messages"]):
                    if not isinstance(msg, HumanMessage):
                        last_ai_message = msg
                        break
                        
            except Exception as e:
                logger.warning(f"Error procesando evento: {e}")
                continue
        
        # Extraer el contenido de la respuesta
        if last_ai_message:
            final_response = last_ai_message.content
        else:
            final_response = "Lo siento, no pude procesar tu mensaje. Por favor intenta nuevamente."
            
    except Exception as e:
        logger.error(f"Error ejecutando workflow: {e}", exc_info=True)
        final_response = "Ocurrió un error al procesar tu mensaje. Por favor intenta más tarde."
    
    return final_response or "Lo siento, no recibí una respuesta válida."


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Maneja mensajes de texto entrantes de usuarios de Telegram.
    
    Args:
        update: Objeto Update de Telegram
        context: Contexto del bot
    """
    # Obtener información del usuario y mensaje
    user = update.effective_user
    message = update.message
    
    if not message or not message.text:
        return
    
    user_id = user.id
    user_message = message.text.strip()
    
    # Crear thread_id único para este usuario
    thread_id = f"telegram_{user_id}"
    
    logger.info(f"Mensaje recibido de usuario {user_id}: {user_message[:100]}")
    
    # Enviar indicador de "escribiendo..."
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        # Procesar el mensaje en el executor (workflow es síncrono)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            process_user_message,
            user_message,
            thread_id
        )
        
        # Enviar la respuesta al usuario
        await message.reply_text(response)
        logger.info(f"Respuesta enviada a usuario {user_id}")
        
    except Exception as e:
        logger.error(f"Error manejando mensaje: {e}", exc_info=True)
        error_message = (
            "Lo siento, ocurrió un error al procesar tu mensaje. "
            "Por favor intenta nuevamente más tarde."
        )
        await message.reply_text(error_message)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Maneja el comando /start.
    
    Args:
        update: Objeto Update de Telegram
        context: Contexto del bot
    """
    welcome_message = (
        "¡Hola! 👋\n\n"
        "Soy tu asistente virtual para el consultorio dental. "
        "Puedo ayudarte con:\n"
        "• Consultas sobre servicios y precios\n"
        "• Agendar citas\n"
        "• Reprogramar o cancelar citas\n\n"
        "¿En qué puedo ayudarte hoy?"
    )
    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Maneja el comando /help.
    
    Args:
        update: Objeto Update de Telegram
        context: Contexto del bot
    """
    help_message = (
        "📚 *Ayuda*\n\n"
        "Puedes escribirme mensajes en texto libre y yo te ayudaré con:\n\n"
        "• *Consultas*: Información sobre servicios, precios, horarios\n"
        "• *Reservas*: Agendar una cita nueva\n"
        "• *Reprogramación*: Cambiar fecha u hora de una cita existente\n"
        "• *Cancelación*: Cancelar una cita\n\n"
        "Solo escribe tu consulta y yo te responderé de inmediato.\n\n"
        "Comandos disponibles:\n"
        "/start - Iniciar conversación\n"
        "/help - Mostrar esta ayuda"
    )
    await update.message.reply_text(help_message, parse_mode="Markdown")


def create_bot_application() -> Application:
    """
    Crea y configura la aplicación del bot de Telegram.
    
    Returns:
        Application configurada del bot
    """
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN no está configurado. "
            "Por favor configúralo en tu archivo .env"
        )
    
    # Crear la aplicación del bot
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Registrar handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Bot de Telegram configurado correctamente")
    
    return application

