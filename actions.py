# actions.py
"""
ActionRouter: Decoupled Intent Execution Layer
- Maps predicted intents to modular handler functions
- Returns structured action responses: { "success": bool, "action": str, "reply": str, "metadata": dict }
- Provides safe mock execution for destructive commands (shutdown, restart)
- Supports external application launching (browser, settings, calculator)
"""

import os
import sys
import time
import random
import webbrowser
import subprocess
from typing import Dict, Any, Callable

class ActionRouter:
    def __init__(self, safe_mode: bool = True):
        self.safe_mode = safe_mode
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self._register_default_handlers()

    def register(self, intent: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Register or override an action handler for a specific intent."""
        self._handlers[intent] = handler

    def dispatch(self, intent: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Dispatch intent to its registered handler with structured output."""
        context = context or {}
        handler = self._handlers.get(intent, self._handle_unknown)
        try:
            t0 = time.perf_counter()
            res = handler(context)
            res["execution_time_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            return res
        except Exception as e:
            return {
                "success": False,
                "action": intent,
                "reply": f"An error occurred while executing {intent}.",
                "error": str(e),
                "metadata": context
            }

    # --------------------------
    # DEFAULT HANDLERS
    # --------------------------
    def _register_default_handlers(self):
        self._handlers["get_time"] = self._handle_time
        self._handlers["get_date"] = self._handle_date
        self._handlers["greeting"] = self._handle_greeting
        self._handlers["tell_joke"] = self._handle_joke
        self._handlers["turn_on_light"] = self._handle_light_on
        self._handlers["turn_off_light"] = self._handle_light_off
        self._handlers["increase_volume"] = self._handle_volume_up
        self._handlers["decrease_volume"] = self._handle_volume_down
        self._handlers["play_music"] = self._handle_play_music
        self._handlers["stop_music"] = self._handle_stop_music
        self._handlers["open_youtube"] = self._handle_open_youtube
        self._handlers["open_google"] = self._handle_open_google
        self._handlers["open_calculator"] = self._handle_open_calculator
        self._handlers["open_settings"] = self._handle_open_settings
        self._handlers["set_alarm"] = self._handle_set_alarm
        self._handlers["set_reminder"] = self._handle_set_reminder
        self._handlers["shutdown_device"] = self._handle_shutdown
        self._handlers["restart_device"] = self._handle_restart
        self._handlers["unknown"] = self._handle_unknown

    def _handle_time(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        current_time = time.strftime("%I:%M %p")
        return {
            "success": True,
            "action": "get_time",
            "reply": f"The current time is {current_time}.",
            "metadata": {"time": current_time}
        }

    def _handle_date(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        current_date = time.strftime("%A, %B %d, %Y")
        return {
            "success": True,
            "action": "get_date",
            "reply": f"Today is {current_date}.",
            "metadata": {"date": current_date}
        }

    def _handle_greeting(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        greetings = [
            "Hello! How can I assist you today?",
            "Hi there! What can I do for you?",
            "Greetings! I'm ready for your commands."
        ]
        return {
            "success": True,
            "action": "greeting",
            "reply": random.choice(greetings),
            "metadata": {}
        }

    def _handle_joke(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "There are 10 types of people in the world: those who understand binary, and those who don't.",
            "Why did the developer go broke? Because he used up all his cache!",
            "A SQL query walks into a bar, walks up to two tables and asks: Can I join you?"
        ]
        return {
            "success": True,
            "action": "tell_joke",
            "reply": random.choice(jokes),
            "metadata": {}
        }

    def _handle_light_on(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "turn_on_light",
            "reply": "Turning on the lights.",
            "metadata": {"state": "on"}
        }

    def _handle_light_off(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "turn_off_light",
            "reply": "Turning off the lights.",
            "metadata": {"state": "off"}
        }

    def _handle_volume_up(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "increase_volume",
            "reply": "Increasing the volume.",
            "metadata": {"step": "+10%"}
        }

    def _handle_volume_down(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "decrease_volume",
            "reply": "Decreasing the volume.",
            "metadata": {"step": "-10%"}
        }

    def _handle_play_music(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "play_music",
            "reply": "Playing music playback.",
            "metadata": {"state": "playing"}
        }

    def _handle_stop_music(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "stop_music",
            "reply": "Stopping music playback.",
            "metadata": {"state": "stopped"}
        }

    def _handle_open_youtube(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if not self.safe_mode:
            webbrowser.open("https://www.youtube.com")
        return {
            "success": True,
            "action": "open_youtube",
            "reply": "Opening YouTube.",
            "metadata": {"url": "https://www.youtube.com"}
        }

    def _handle_open_google(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if not self.safe_mode:
            webbrowser.open("https://www.google.com")
        return {
            "success": True,
            "action": "open_google",
            "reply": "Opening Google Search.",
            "metadata": {"url": "https://www.google.com"}
        }

    def _handle_open_calculator(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if not self.safe_mode and sys.platform.startswith("win"):
            subprocess.Popen("calc.exe", shell=True)
        return {
            "success": True,
            "action": "open_calculator",
            "reply": "Launching the calculator.",
            "metadata": {}
        }

    def _handle_open_settings(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if not self.safe_mode and sys.platform.startswith("win"):
            subprocess.Popen("start ms-settings:", shell=True)
        return {
            "success": True,
            "action": "open_settings",
            "reply": "Opening system settings.",
            "metadata": {}
        }

    def _handle_set_alarm(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "set_alarm",
            "reply": "Alarm has been scheduled.",
            "metadata": {}
        }

    def _handle_set_reminder(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "set_reminder",
            "reply": "Reminder has been set.",
            "metadata": {}
        }

    def _handle_shutdown(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        # CRITICAL INTERVIEW REQUIREMENT: Always mock destructive actions!
        return {
            "success": True,
            "action": "shutdown_device",
            "reply": "Shutdown command recognized. System shutdown prevented in demonstration mode.",
            "metadata": {"simulated": True}
        }

    def _handle_restart(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        # CRITICAL INTERVIEW REQUIREMENT: Always mock destructive actions!
        return {
            "success": True,
            "action": "restart_device",
            "reply": "Restart command recognized. System reboot prevented in demonstration mode.",
            "metadata": {"simulated": True}
        }

    def _handle_unknown(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": False,
            "action": "unknown",
            "reply": "I'm not sure how to help with that yet. I can check time, volume, play music, or open apps.",
            "metadata": {}
        }
