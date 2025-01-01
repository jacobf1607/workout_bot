from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update
import logging
import tempfile
from pathlib import Path
import requests
from datetime import datetime
import json
import traceback
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np
import io
import asyncio
import pandas as pd
import sys
import matplotlib.dates as mdates
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY_1')

print(OPENAI_API_KEY)

class WorkoutBot:
    def __init__(self):
        self.TELEGRAM_TOKEN = TELEGRAM_TOKEN
        self.OPENAI_API_KEY = OPENAI_API_KEY
        
        # Setup logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler('bot.log'),
                logging.StreamHandler()
            ]
        )
        
        try:
            # Initialize MongoDB connection
            self.mongo_client = MongoClient(MONGODB_URI)
            self.db = self.mongo_client.workout_tracker
            
            # Create indexes for better query performance
            self.db.users.create_index("user_id", unique=True)
            self.db.workouts.create_index("user_id")
            self.db.workouts.create_index([("user_id", 1), ("date", -1)])
            self.db.workouts.create_index([("user_id", 1), ("muscle_group", 1)])
            self.db.workouts.create_index([("user_id", 1), ("exercise", 1)])
            self.db.workouts.create_index([("user_id", 1), ("normalized_exercise", 1)])
            self.db.workouts.create_index("created_at")
            
            # Test the connection
            self.db.command('ping')
            logging.info("Successfully connected to MongoDB")
            
        except Exception as e:
            logging.error(f"Error initializing MongoDB: {str(e)}\n{traceback.format_exc()}")
            raise Exception("Failed to initialize MongoDB connection")




    def save_user_info(self, user):
        """Save or update user information in MongoDB"""
        try:
            user_data = {
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "updated_at": datetime.now()
            }
            
            # Use upsert to either insert new or update existing
            self.db.users.update_one(
                {"user_id": user.id},
                {"$set": user_data},
                upsert=True
            )
            
            logging.info(f"User info saved successfully for user {user.id}")
            return True
        except Exception as e:
            logging.error(f"Error saving user info: {str(e)}\n{traceback.format_exc()}")
            return False

    def get_user_info(self, user_id):
        """Get user information from MongoDB"""
        try:
            user = self.db.users.find_one({"user_id": user_id})
            return user
        except Exception as e:
            logging.error(f"Error getting user info: {str(e)}\n{traceback.format_exc()}")
            return None


    def get_recent_workouts(self, user_id, days):
        """Get workouts from the last X days"""
        try:
            # Calculate the date threshold
            threshold_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Query MongoDB
            workouts = self.db.workouts.find({
                "user_id": user_id,
                "date": {"$gte": threshold_date}
            }).sort("date", -1)
            
            return list(workouts)
        except Exception as e:
            logging.error(f"Error getting recent workouts: {str(e)}")
            return []

    def get_workout_data_for_graph(self, user_id, exercise=None, muscle_group=None, days=None):
        """Get workout data formatted for graphing"""
        try:
            # Build query conditions
            query = {"user_id": user_id}
            
            if exercise:
                query["exercise"] = exercise
            if muscle_group:
                query["muscle_group"] = muscle_group
            if days:
                threshold_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                query["date"] = {"$gte": threshold_date}
            
            # Query MongoDB and sort by date
            workouts = self.db.workouts.find(query).sort("date", 1)
            
            return list(workouts)
        except Exception as e:
            logging.error(f"Error getting graph data: {str(e)}")
            return []
        


    def get_user_muscle_groups(self, user_id):
        """Get unique muscle groups for a user"""
        try:
            muscle_groups = self.db.workouts.distinct(
                "muscle_group",
                {
                    "user_id": user_id,
                    "muscle_group": {"$ne": None}
                }
            )
            return sorted(muscle_groups)
        except Exception as e:
            logging.error(f"Error getting muscle groups: {str(e)}")
            return []

    def get_user_exercises(self, user_id):
        """Get unique exercises for a user"""
        try:
            exercises = self.db.workouts.distinct(
                "exercise",
                {
                    "user_id": user_id,
                    "exercise": {"$ne": None}
                }
            )
            return sorted(exercises)
        except Exception as e:
            logging.error(f"Error getting exercises: {str(e)}")
            return []
            

            
    def get_exercise_workouts(self, user_id, exercise):
        """Get workouts for a specific exercise"""
        try:
            workouts = self.db.workouts.find({
                "user_id": user_id,
                "exercise": exercise
            }).sort("date", -1)
            
            return list(workouts)
        except Exception as e:
            logging.error(f"Error getting exercise workouts: {str(e)}")
            return []

    def format_workout_history(self, workouts, title="Your Workout History"):
        """Format workout history for display"""
        if not workouts:
            return "No workout history found."
            
        message = f"📋 {title}\n\n"
        
        for workout in workouts:
            message += f"📅 {workout['date']}\n"
            message += f"💪 {workout['muscle_group']} - {workout['exercise']}\n"
            
            if workout['weight']:
                message += f"⚖️ Weight: {workout['weight']} kg\n"
            if workout['reps']:
                message += f"🔄 Reps: {workout['reps']}\n"
            if workout['sets']:
                message += f"📊 Sets: {workout['sets']}\n"
            if workout['notes']:
                message += f"📝 Notes: {workout['notes']}\n"
            
            message += "\n"
            
        return message


    def format_workout_message(self, workout, include_header=True):
        """Format workout information consistently"""
        message = ""
        if include_header:
            message += "✅ Workout Updated!\n"
        
        message += (
            f"📆 Date: {workout['date']}\n"
            f"💪 Muscle Group: {workout.get('muscle_group', 'Not specified')}\n"
            f"🎯 Exercise: {workout.get('exercise', 'Not specified')}\n"
        )
        
        if workout.get('weight'):
            message += f"⚖️ Weight: {workout['weight']} kg\n"
        if workout.get('reps'):
            message += f"🔄 Reps: {workout['reps']}\n"
        if workout.get('sets'):
            message += f"📊 Sets: {workout['sets']}\n"
        if workout.get('notes'):
            message += f"📝 Notes: {workout['notes']}\n"
        
        return message
                
    def log_workout(self, workout_info):
        """Log workout to MongoDB with standardized formatting"""
        try:
            # Get and normalize exercise name and muscle group
            exercise = self.normalize_name(workout_info.get('exercise')) if workout_info.get('exercise') else None
            muscle_group = self.normalize_name(workout_info.get('muscle_group')) if workout_info.get('muscle_group') else None
            
            workout_doc = {
                "user_id": workout_info['user_id'],
                "date": workout_info['date'],
                "muscle_group": muscle_group,
                "exercise": exercise,
                "weight": workout_info.get('weight'),
                "reps": workout_info.get('reps'),
                "sets": workout_info.get('sets'),
                "notes": workout_info.get('notes'),
                "audio_file": workout_info.get('audio_file'),
                "created_at": datetime.now()
            }
            
            # Insert the workout document
            result = self.db.workouts.insert_one(workout_doc)
            
            logging.info(f"Workout logged successfully for user {workout_info['user_id']}")
            return True
        except Exception as e:
            logging.error(f"Error logging workout: {str(e)}\n{traceback.format_exc()}")
            return False
        
    async def start(self, update, context):
        """Handle /start command"""
        try:
            user = update.effective_user
            self.save_user_info(user)
            
            await update.message.reply_text(
                f"🏋️‍♂️ Welcome {user.first_name}!\n\n"
                "Send me a voice message describing your workout.\n\n"
                "Format: Mention muscle group, exercise, weight, reps, and sets.\n"
                "Example: 'Chest workout, bench press, 100kgs, 3 sets of 8 reps'\n\n"
                "Available Commands:\n"
                "/history - View your workout history and progress graphs\n"
                "/export - Download your workout history as Excel file\n"
                "/stats - View your workout statistics and achievements 📊\n"
                "/erase - Delete all your workout history ⚠️\n"
                "/merge - Combine duplicate exercise names 🔄\n"
                "/help - Show all commands and detailed usage info ℹ️\n\n"
                "Tips:\n"
                "• Voice messages are automatically processed\n"
                "• Weights in pounds are converted to kilograms\n"
                "• You can edit or delete workouts after logging them\n"
                "• Progress graphs are available for each exercise\n"
                "• Use /help anytime to see all available commands"
            )
            logging.info(f"Start command processed for user {user.id}")
        except Exception as e:
            logging.error(f"Error in start command: {str(e)}\n{traceback.format_exc()}")
            await update.message.reply_text("❌ An error occurred. Please try again.")

    def normalize_name(self, name):
        """
        Normalize any name (exercise or muscle group):
        - Convert to lowercase
        - Remove special characters
        - Remove extra spaces
        """
        if not name:
            return None
        
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove special characters (keep only letters, numbers and spaces)
        normalized = ''.join(char for char in normalized if char.isalnum() or char.isspace())
        
        # Remove extra spaces and trim
        normalized = ' '.join(normalized.split())
        
        return normalized

    def fuzzy_match_names(self, name1, name2, threshold=0.8):
        """Fuzzy match two names using Levenshtein distance"""
        from difflib import SequenceMatcher
        
        if not name1 or not name2:
            return False
            
        # Normalize both names
        name1 = self.normalize_name(name1)
        name2 = self.normalize_name(name2)
        
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, name1, name2).ratio()
        
        return similarity >= threshold

    # Then update the send_workout_graph method - replace the date formatting section with this:
    async def history(self, update, context):
        """Handle /history command - Show all muscle groups"""
        try:
            user = update.effective_user
            muscle_groups = self.get_user_muscle_groups(user.id)
            
            if not muscle_groups:
                await update.message.reply_text("No workout history found. Start logging your workouts!")
                return
                
            keyboard = []
            # Create a button for each muscle group
            for muscle in muscle_groups:
                keyboard.append([InlineKeyboardButton(f"💪 {muscle}", callback_data=f"select_muscle_{muscle}")])
                
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "Select a muscle group to view exercises:",
                reply_markup=reply_markup
            )
        except Exception as e:
            logging.error(f"Error in history command: {str(e)}\n{traceback.format_exc()}")
            await update.message.reply_text("❌ An error occurred. Please try again.")


    def get_muscle_group_exercises(self, user_id, muscle_group):
        """Get unique exercises for a specific muscle group"""
        try:
            exercises = self.db.workouts.distinct(
                "exercise",
                {
                    "user_id": user_id,
                    "muscle_group": muscle_group,
                    "exercise": {"$ne": None}
                }
            )
            return sorted(exercises)
        except Exception as e:
            logging.error(f"Error getting muscle group exercises: {str(e)}")
            return []

    def get_last_five_workouts(self, user_id, exercise):
        """Get the last 5 workouts for a specific exercise"""
        try:
            workouts = self.db.workouts.find({
                "user_id": user_id,
                "exercise": exercise
            }).sort("date", -1).limit(5)
            
            return list(workouts)
        except Exception as e:
            logging.error(f"Error getting last five workouts: {str(e)}")
            return []

    def get_all_user_workouts(self, user_id):
        """Get all workouts for a specific user"""
        try:
            # Query MongoDB and sort by date and created_at
            workouts = self.db.workouts.find(
                {"user_id": user_id}
            ).sort([("date", -1), ("created_at", -1)])
            
            return list(workouts)
        except Exception as e:
            logging.error(f"Error getting all workouts: {str(e)}")
            return []

    async def export_workouts(self, update, context):
        """Handle /export command - Generate and send Excel file with workout history"""
        try:
            user = update.effective_user
            workouts = self.get_all_user_workouts(user.id)
            
            if not workouts:
                await update.message.reply_text("No workout history found to export.")
                return
            
            # Process workouts to handle ObjectId
            processed_workouts = []
            for workout in workouts:
                workout_dict = {
                    'date': workout['date'],
                    'muscle_group': workout.get('muscle_group', ''),
                    'exercise': workout.get('exercise', ''),
                    'weight': workout.get('weight', ''),
                    'reps': workout.get('reps', ''),
                    'sets': workout.get('sets', ''),
                    'notes': workout.get('notes', ''),
                    'created_at': workout.get('created_at', '')
                }
                processed_workouts.append(workout_dict)
            
            # Create Excel file using pandas
            df = pd.DataFrame(processed_workouts)
            
            # Reorder columns
            columns = ['date', 'muscle_group', 'exercise', 'weight', 'reps', 'sets', 'notes', 'created_at']
            df = df[columns]
            
            # Rename columns for better readability
            df.columns = ['Date', 'Muscle Group', 'Exercise', 'Weight (kg)', 'Reps', 'Sets', 'Notes', 'Created At']
            
            # Create a BytesIO object to store the Excel file
            excel_file = io.BytesIO()
            
            # Create Excel writer object with xlsxwriter engine
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Workout History', index=False)
                
                # Get the workbook and worksheet objects
                workbook = writer.book
                worksheet = writer.sheets['Workout History']
                
                # Add some formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'bg_color': '#D9D9D9',
                    'border': 1
                })
                
                # Write headers with formatting
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    
                # Set column widths
                worksheet.set_column('A:A', 12)  # Date
                worksheet.set_column('B:B', 15)  # Muscle Group
                worksheet.set_column('C:C', 20)  # Exercise
                worksheet.set_column('D:F', 12)  # Weight, Reps, Sets
                worksheet.set_column('G:G', 30)  # Notes
                worksheet.set_column('H:H', 20)  # Created At
            
            # Reset file pointer
            excel_file.seek(0)
            
            # Generate filename with current date
            current_date = datetime.now().strftime('%Y-%m-%d')
            filename = f"workout_history_{current_date}.xlsx"
            
            # Send the file
            await update.message.reply_document(
                document=InputFile(excel_file, filename=filename),
                caption="📊 Here's your complete workout history!\n\nIncludes all your workouts with dates, exercises, and details."
            )
            
        except Exception as e:
            logging.error(f"Error exporting workouts: {str(e)}\n{traceback.format_exc()}")
            await update.message.reply_text("❌ An error occurred while exporting your workout history. Please try again.")
                
    async def handle_history_callback(self, update, context):
        """Handle callback queries from history menu"""
        try:
            callback_query = update.callback_query
            user_id = callback_query.from_user.id
            data = callback_query.data

            await callback_query.answer()

            if data.startswith("select_muscle_"):
                # When a muscle group is selected, show its exercises
                muscle_group = data.replace("select_muscle_", "")
                exercises = self.get_muscle_group_exercises(user_id, muscle_group)
                
                if not exercises:
                    await callback_query.edit_message_text(
                        f"No exercises found for {muscle_group}. Start logging your workouts!"
                    )
                    return
                    
                keyboard = []
                for exercise in exercises:
                    keyboard.append([
                        InlineKeyboardButton(f"🎯 {exercise}", callback_data=f"select_exercise_{exercise}")
                    ])
                keyboard.append([InlineKeyboardButton("🔙 Back to Muscle Groups", callback_data="history_menu")])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                await callback_query.edit_message_text(
                    f"Select an exercise from {muscle_group}:",
                    reply_markup=reply_markup
                )

            elif data.startswith("select_exercise_"):
                # When an exercise is selected, show last 5 workouts and graph
                exercise = data.replace("select_exercise_", "")
                workouts = self.get_last_five_workouts(user_id, exercise)
                
                if not workouts:
                    await callback_query.edit_message_text(
                        f"No workout history found for {exercise}."
                    )
                    return

                # First send the text summary
                message = self.format_workout_history(workouts, f"Last 5 {exercise} Workouts")
                await callback_query.edit_message_text(message)

                # Then send the graph
                graph_data = self.get_workout_data_for_graph(user_id, exercise=exercise)
                await self.send_workout_graph(update, context, graph_data, f"Progress for {exercise}")

            elif data == "history_menu":
                # Return to main muscle groups menu
                muscle_groups = self.get_user_muscle_groups(user_id)
                keyboard = []
                for muscle in muscle_groups:
                    keyboard.append([InlineKeyboardButton(f"💪 {muscle}", callback_data=f"select_muscle_{muscle}")])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                await callback_query.edit_message_text(
                    "Select a muscle group to view exercises:",
                    reply_markup=reply_markup
                )

            elif data.startswith("confirm_delete_"):
                # Handle workout deletion
                workout_id = data.replace("confirm_delete_", "")  # Keep as string for ObjectId
                if self.delete_workout(workout_id):
                    await callback_query.edit_message_text("✅ Workout deleted successfully!")
                else:
                    await callback_query.edit_message_text("❌ Failed to delete workout.")

            elif data == "cancel_delete":
                await callback_query.edit_message_text("Deletion cancelled.")

            elif data.startswith("edit_field_"):
                # Store the field being edited in user_data
                parts = data.split("_")
                field = parts[2]
                workout_id = parts[3]  # Keep as string for ObjectId
                workout = self.get_workout_by_id(workout_id)
                
                if not workout:
                    await callback_query.edit_message_text("Workout not found.")
                    return

                # Show the current workout info and the field being edited
                message = self.format_workout_message(workout, include_header=False)
                current_value = workout.get(field, 'Not set')
                message += f"\nEditing {field.replace('_', ' ').title()}\n"
                message += f"Current value: {current_value}\n\n"
                message += f"Please send the new value for {field.replace('_', ' ').title()}:"

                context.user_data['editing'] = {
                    'field': field,
                    'workout_id': workout_id
                }
                
                await callback_query.edit_message_text(message)

            elif data == "cancel_edit":
                if 'editing' in context.user_data:
                    del context.user_data['editing']
                # Get the latest workout to show final values
                workout = self.get_latest_workout(user_id)
                if workout:
                    # Show final message with success header
                    message = self.format_workout_message(workout, include_header=True)
                    await callback_query.edit_message_text(message)
                else:
                    await callback_query.edit_message_text("✅ Updates completed!")

        except Exception as e:
            logging.error(f"Error in callback query: {str(e)}\n{traceback.format_exc()}")
            await callback_query.edit_message_text("❌ An error occurred. Please try again.")


    async def send_workout_graph(self, update, context, workouts, title):
            """Generate and send a graph visualization as an image"""
            if not workouts:
                await update.callback_query.edit_message_text("No workout data available for graphing.")
                return

            try:
                # Convert data to arrays - reverse the lists so newest is last (right side of graph)
                weights = [w['weight'] or 0 for w in reversed(workouts)]
                reps = [w['reps'] or 0 for w in reversed(workouts)]
                sets = [w['sets'] or 0 for w in reversed(workouts)]
                
                # Create x-axis points (1 through 5, representing last 5 workouts)
                x_points = list(range(1, len(workouts) + 1))

                # Create figure
                fig, ax1 = plt.subplots(figsize=(10, 6))
                plt.suptitle(title, fontsize=16, y=1.05)

                # Set up the second y-axis sharing the same x-axis
                ax2 = ax1.twinx()

                # Plot data
                # Orange line for weight (on right y-axis)
                line1 = ax2.plot(x_points, weights, color='#FF9800', marker='o', linewidth=2, 
                            label='Weight (kg)', markersize=8)[0]
                
                # Green line for reps (on left y-axis)
                line2 = ax1.plot(x_points, reps, color='#4CAF50', marker='s', linewidth=2, 
                            label='Reps', markersize=8)[0]
                
                # Blue line for sets (on left y-axis)
                line3 = ax1.plot(x_points, sets, color='#2196F3', marker='^', linewidth=2, 
                            label='Sets', markersize=8)[0]

                # Customize axes
                ax1.set_xlabel('Last 5 Workouts (1 = oldest)', fontsize=10)
                ax1.set_ylabel('Reps and Sets', color='black', fontsize=10)
                ax2.set_ylabel('Weight (kg)', color='black', fontsize=10)

                # Set x-axis ticks
                ax1.set_xticks(x_points)
                
                # Add grid
                ax1.grid(True, alpha=0.3)

                # Combine legends from both axes
                lines = [line1, line2, line3]
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, -0.15),
                        ncol=3, frameon=False)

                # Adjust layout
                plt.tight_layout()

                # Save plot to bytes buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                buf.seek(0)
                plt.close(fig)

                # Send image with caption
                caption = (
                    f"📊 {title}\n\n"
                    "🟠 Orange Line: Weight (kg)\n"
                    "🟢 Green Line: Reps\n"
                    "🔵 Blue Line: Sets\n\n"
                    "⬅️ Oldest workout  |  Newest workout ➡️"
                )

                await context.bot.send_photo(
                    chat_id=update.callback_query.message.chat_id,
                    photo=InputFile(buf, 'workout_progress.png'),
                    caption=caption
                )

            except Exception as e:
                logging.error(f"Error sending graph: {str(e)}\n{traceback.format_exc()}")
                await update.callback_query.edit_message_text(
                    "❌ An error occurred while generating the graph. Please try again."
                )

    async def handle_voice(self, update, context):
        """Process voice messages"""
        try:
            user = update.effective_user
            user_info = self.get_user_info(user.id)
            
            if not user_info:
                self.save_user_info(user)
                user_info = self.get_user_info(user.id)

            voice = update.message.voice
            file = await context.bot.get_file(voice.file_id)
            temp_file = Path(tempfile.gettempdir()) / f"{voice.file_id}.ogg"
            await file.download_to_drive(temp_file)
            
            status_msg = await update.message.reply_text("🎯 Processing your workout...")
            
            transcription = self.transcribe_audio(temp_file)
            if not transcription:
                await status_msg.edit_text("❌ Failed to transcribe audio. Please try again.")
                return

            workout_info = self.parse_workout_info(transcription)
            if not workout_info:
                await status_msg.edit_text("❌ Couldn't understand workout details. Please try again.")
                return

            workout_info.update({
                'user_id': user.id,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'audio_file': f"voice_{voice.file_id}"
            })

            if self.log_workout(workout_info):
                if user_info['username']:
                    display_name = user_info['username']
                elif user_info['first_name']:
                    if user_info['last_name']:
                        display_name = f"{user_info['first_name']} {user_info['last_name']}"
                    else:
                        display_name = user_info['first_name']
                else:
                    display_name = f"User {user_info['user_id']}"
                
                summary = (
                    f"✅ Workout logged!\n\n"
                    f"👤 User: @{display_name}\n"
                    f"📆 Date: {workout_info['date']}\n"
                    f"💪 Muscle Group: {workout_info.get('muscle_group', 'Not specified')}\n"
                    f"🎯 Exercise: {workout_info.get('exercise', 'Not specified')}\n"
                )
                
                if workout_info.get('weight'):
                    summary += f"⚖️ Weight: {workout_info['weight']} kg\n"
                if workout_info.get('reps'):
                    summary += f"🔄 Reps: {workout_info['reps']}\n"
                if workout_info.get('sets'):
                    summary += f"📊 Sets: {workout_info['sets']}\n"
                if workout_info.get('notes'):
                    summary += f"\n📝 Notes: {workout_info['notes']}"
                
                keyboard = [
                    [
                        InlineKeyboardButton("✏️ Edit", callback_data=f"edit_workout_latest"),
                        InlineKeyboardButton("🗑️ Delete", callback_data=f"delete_workout_latest")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await status_msg.edit_text(summary, reply_markup=reply_markup)

            else:
                await status_msg.edit_text("❌ Failed to log workout. Please try again.")
            
            temp_file.unlink(missing_ok=True)
            
        except Exception as e:
            logging.error(f"Error processing voice message: {str(e)}\n{traceback.format_exc()}")
            await update.message.reply_text("❌ An error occurred. Please try again.")

    def transcribe_audio(self, audio_file_path):
        """Transcribe audio using OpenAI's Whisper API"""
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.OPENAI_API_KEY}"}
        
        try:
            with open(audio_file_path, "rb") as audio_file:
                files = {
                    "file": ("audio.ogg", audio_file, "audio/ogg"),
                    "model": (None, "whisper-1"),
                    "language": (None, "en"),  # Specify English language
                    "response_format": (None, "text")  # Get plain text response
                }
                response = requests.post(url, headers=headers, files=files)
            
            if response.status_code == 200:
                return response.text
            else:
                logging.error(f"Transcription API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logging.error(f"Transcription error: {str(e)}\n{traceback.format_exc()}")
            return None

    def get_user_stats(self, user_id):
        """Get user's workout statistics"""
        try:
            # Get basic stats using aggregation
            pipeline = [
                {"$match": {"user_id": user_id}},
                {"$group": {
                    "_id": None,
                    "total_workouts": {"$sum": 1},
                    "total_days": {"$addToSet": "$date"},
                    "muscle_groups": {"$addToSet": "$muscle_group"},
                    "exercises": {"$addToSet": "$exercise"},
                    "first_workout": {"$min": "$date"},
                    "last_workout": {"$max": "$date"}
                }}
            ]
            
            stats_result = list(self.db.workouts.aggregate(pipeline))
            if not stats_result:
                return None
                
            stats = stats_result[0]
            
            # Format the stats
            stats_formatted = {
                "total_workouts": stats["total_workouts"],
                "total_days": len(stats["total_days"]),
                "muscle_groups": len(stats["muscle_groups"]),
                "exercises": len(stats["exercises"]),
                "first_workout": stats["first_workout"],
                "last_workout": stats["last_workout"]
            }
            
            # Get max weights for each exercise
            max_weights_pipeline = [
                {"$match": {
                    "user_id": user_id,
                    "weight": {"$ne": None}
                }},
                {"$sort": {"weight": -1}},
                {"$group": {
                    "_id": "$exercise",
                    "max_weight": {"$first": "$weight"},
                    "date": {"$first": "$date"},
                    "muscle_group": {"$first": "$muscle_group"}
                }},
                {"$sort": {"max_weight": -1}}
            ]
            
            max_weights = list(self.db.workouts.aggregate(max_weights_pipeline))
            
            if max_weights:
                stats_formatted["max_weights"] = [
                    {
                        "exercise": w["_id"],
                        "muscle_group": w["muscle_group"],
                        "max_weight": w["max_weight"],
                        "date": w["date"]
                    }
                    for w in max_weights
                ]
            
            return stats_formatted
            
        except Exception as e:
            logging.error(f"Error getting user stats: {str(e)}")
            return None

    async def stats(self, update, context):
        """Handle /stats command - Show user statistics"""
        try:
            user = update.effective_user
            stats = self.get_user_stats(user.id)
            
            if not stats or stats['total_workouts'] == 0:
                await update.message.reply_text(
                    "No workout history found yet. Start logging your workouts!"
                )
                return
            
            # Calculate days between first and last workout
            first_date = datetime.strptime(stats['first_workout'], '%Y-%m-%d')
            last_date = datetime.strptime(stats['last_workout'], '%Y-%m-%d')
            days_tracking = (last_date - first_date).days + 1
            
            message = (
                "📊 Your Workout Statistics\n\n"
                f"🏋️‍♂️ Total Workouts: {stats['total_workouts']}\n"
                f"📅 Active Days: {stats['total_days']}\n"
                f"💪 Muscle Groups Trained: {stats['muscle_groups']}\n"
                f"🎯 Different Exercises: {stats['exercises']}\n"
                f"⏱ Days Tracking: {days_tracking}\n"
                f"📆 First Workout: {stats['first_workout']}\n"
                f"🔄 Last Workout: {stats['last_workout']}\n\n"
            )
            
            if 'max_weights' in stats and stats['max_weights']:
                message += "🏆 Personal Records (Heaviest Lifts):\n\n"
                for record in stats['max_weights']:
                    message += (
                        f"💪 {record['exercise']} ({record['muscle_group']})\n"
                        f"⚖️ Weight: {record['max_weight']} kg\n"
                        f"📅 Date: {record['date']}\n\n"
                    )
            
            await update.message.reply_text(message)
            
        except Exception as e:
            logging.error(f"Error in stats command: {str(e)}\n{traceback.format_exc()}")
            await update.message.reply_text("❌ An error occurred. Please try again.")

    def get_similar_names(self, user_id, name_type='exercise'):
        """Get similar exercise or muscle group names"""
        try:
            field = 'exercise' if name_type == 'exercise' else 'muscle_group'
            
            pipeline = [
                {"$match": {
                    "user_id": user_id,
                    field: {"$ne": None}
                }},
                {"$group": {
                    "_id": f"${field}",
                    "usage_count": {"$sum": 1}
                }},
                {"$match": {
                    "usage_count": {"$gt": 1}
                }},
                {"$sort": {"usage_count": -1}}
            ]
            
            results = list(self.db.workouts.aggregate(pipeline))
            
            # Now use fuzzy matching to group similar names
            similar_groups = []
            processed_names = set()
            
            for item in results:
                name = item["_id"]
                if name in processed_names:
                    continue
                    
                # Find all fuzzy matches for this name
                matches = []
                count = 0
                for other_item in results:
                    other_name = other_item["_id"]
                    if other_name not in processed_names and self.fuzzy_match_names(name, other_name):
                        matches.append(other_name)
                        count += other_item["usage_count"]
                        processed_names.add(other_name)
                
                if len(matches) > 1:  # Only add groups with actual similar names
                    similar_groups.append({
                        "names": matches,
                        "usage_count": count
                    })
            
            return similar_groups
                
        except Exception as e:
            logging.error(f"Error getting similar names: {str(e)}")
            return []
        
    def merge_exercises(self, user_id, target_exercise, exercises_to_merge):
        """Merge similar exercises into one"""
        try:
            normalized_target = self.normalize_name(target_exercise)  # Instead of normalize_exercise_name

            
            # Update all workouts with the target exercise name
            result = self.db.workouts.update_many(
                {
                    "user_id": user_id,
                    "normalized_exercise": {"$in": exercises_to_merge}
                },
                {
                    "$set": {
                        "exercise": target_exercise,
                        "normalized_exercise": normalized_target,
                        "display_exercise": target_exercise
                    }
                }
            )
            
            # Log the merges in a separate collection if needed
            merge_records = [
                {
                    "user_id": user_id,
                    "original_exercise": exercise,
                    "merged_into": target_exercise,
                    "merged_at": datetime.now()
                }
                for exercise in exercises_to_merge
            ]
            if merge_records:
                self.db.exercise_merges.insert_many(merge_records)
            
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Error merging exercises: {str(e)}")
            return False
            
    async def merge(self, update, context):
        """Handle /merge command"""
        try:
            user = update.effective_user
            keyboard = [
                [InlineKeyboardButton("Merge Exercises", callback_data="merge_type_exercise")],
                [InlineKeyboardButton("Merge Muscle Groups", callback_data="merge_type_muscle")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                "What would you like to merge?",
                reply_markup=reply_markup
            )
                
        except Exception as e:
            logging.error(f"Error in merge command: {str(e)}")
            await update.message.reply_text("❌ An error occurred. Please try again.")

    async def handle_merge_callback(self, update, context):
        """Handle merge command callbacks"""
        try:
            query = update.callback_query
            user_id = query.from_user.id
            data = query.data
            
            await query.answer()

            if data.startswith("merge_type_"):
                merge_type = data.replace("merge_type_", "")
                similar_names = self.get_similar_names(user_id, merge_type)
                
                if not similar_names:
                    await query.edit_message_text(
                        f"No similar {merge_type} names found to merge."
                    )
                    return
                    
                message = f"Found similar {merge_type} names:\n\n"
                for group in similar_names:
                    message += "• " + ", ".join(group["names"]) + "\n"
                
                await query.edit_message_text(message)
        except Exception as e:
            logging.error(f"Error in merge callback: {str(e)}")
            await query.edit_message_text("❌ An error occurred. Please try again.")

    async def help(self, update, context):
        """Handle /help command - Show all available commands and usage info"""
        try:
            help_text = (
                "🏋️‍♂️ Available Commands:\n\n"
                "/history - View your workout history and progress graphs\n"
                "• See your workouts organized by muscle groups\n"
                "• View progress graphs for each exercise\n"
                "• Track your improvements over time\n\n"
                
                "/stats - View your workout statistics 📊\n"
                "• See total workouts and active days\n"
                "• View personal records for all exercises\n"
                "• Track your heaviest lifts\n\n"
                
                "/export - Download workout history 📥\n"
                "• Get your complete workout history as Excel file\n"
                "• Includes all details and dates\n\n"

                "/merge - Manage duplicate exercises 🔄\n"
                "• Find similar exercise names\n"
                "• Combine duplicates into one\n"
                "• Keep workout history organized\n\n"
                
                "/erase - Delete workout history ⚠️\n"
                "• Completely removes all your workout data\n"
                "• Requires confirmation\n"
                "• Cannot be undone\n\n"
                
                "Voice Messages 🎤\n"
                "• Record your workout details\n"
                "• Include: muscle group, exercise, weight, reps, and sets\n"
                "• Example: 'Chest workout, bench press, 135 pounds, 3 sets of 8 reps'\n\n"
                
                "Tips 💡\n"
                "• Weights in pounds are automatically converted to kilograms\n"
                "• You can edit or delete workouts after logging them\n"
                "• Use /stats regularly to track your progress\n"
                "• Check /history to view exercise-specific graphs"
            )
            
            await update.message.reply_text(help_text)
            
        except Exception as e:
            logging.error(f"Error in help command: {str(e)}\n{traceback.format_exc()}")
            await update.message.reply_text("❌ An error occurred. Please try again.")

    def get_latest_workout(self, user_id):
        """Get the most recent workout for a user"""
        try:
            workout = self.db.workouts.find_one(
                {"user_id": user_id},
                sort=[("created_at", -1)]
            )
            return workout
        except Exception as e:
            logging.error(f"Error getting latest workout: {str(e)}")
            return None
                
    def get_workout_by_id(self, workout_id):
        """Get a specific workout by ID"""
        try:
            workout = self.db.workouts.find_one({"_id": ObjectId(workout_id)})
            if workout:
                # Convert ObjectId to string for easier handling
                workout['id'] = str(workout['_id'])
            return workout
        except Exception as e:
            logging.error(f"Error getting workout by ID: {str(e)}")
            return None
        
    def delete_workout(self, workout_id):
        """Delete a workout by ID"""
        try:
            result = self.db.workouts.delete_one({"_id": ObjectId(workout_id)})
            return result.deleted_count > 0
        except Exception as e:
            logging.error(f"Error deleting workout: {str(e)}")
            return False
            
    def update_workout(self, workout_id, field, value):
        """Update a specific field of a workout"""
        try:
            result = self.db.workouts.update_one(
                {"_id": ObjectId(workout_id)},
                {"$set": {field: value}}
            )
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Error updating workout: {str(e)}")
            return False


    async def handle_edit_callback(self, update, context):
        """Handle edit workout callback"""
        query = update.callback_query
        user_id = query.from_user.id
        
        if query.data == "edit_workout_latest":
            workout = self.get_latest_workout(user_id)
            if not workout:
                await query.edit_message_text("Workout not found.")
                return

            # Show current values without success header
            message = self.format_workout_message(workout, include_header=False)
            message += "\nSelect what you want to edit:"

            # Convert MongoDB's _id to string for the callback data
            workout_id = str(workout['_id'])

            keyboard = [
                [InlineKeyboardButton("💪 Muscle Group", callback_data=f"edit_field_muscle_group_{workout_id}")],
                [InlineKeyboardButton("🎯 Exercise", callback_data=f"edit_field_exercise_{workout_id}")],
                [InlineKeyboardButton("⚖️ Weight", callback_data=f"edit_field_weight_{workout_id}")],
                [InlineKeyboardButton("🔄 Reps", callback_data=f"edit_field_reps_{workout_id}")],
                [InlineKeyboardButton("📊 Sets", callback_data=f"edit_field_sets_{workout_id}")],
                [InlineKeyboardButton("📝 Notes", callback_data=f"edit_field_notes_{workout_id}")],
                [InlineKeyboardButton("✅ Done", callback_data="cancel_edit")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(message, reply_markup=reply_markup)
            return

        # Handle edit field selections
        elif query.data.startswith("edit_field_"):
            field = query.data.split("_")[2]
            workout_id = query.data.split("_")[3]  # This is now a string of the ObjectId
            workout = self.get_workout_by_id(workout_id)
            
            if not workout:
                await query.edit_message_text("Workout not found.")
                return

            # Show the current workout info and the field being edited
            message = self.format_workout_message(workout, include_header=False)
            current_value = workout.get(field, 'Not set')
            message += f"\nEditing {field.replace('_', ' ').title()}\n"
            message += f"Current value: {current_value}\n\n"
            message += f"Please send the new value for {field.replace('_', ' ').title()}:"

            context.user_data['editing'] = {
                'field': field,
                'workout_id': workout_id
            }

            await query.edit_message_text(message)

    async def handle_edit_text(self, update, context):
        """Handle text messages for editing workout fields"""
        try:
            if 'editing' not in context.user_data:
                return

            editing = context.user_data['editing']
            field = editing['field']
            workout_id = editing['workout_id']  # This is a string of the ObjectId
            new_value = update.message.text.strip()

            # Convert value to appropriate type based on field
            if field in ['weight', 'reps', 'sets']:
                try:
                    new_value = float(new_value) if field == 'weight' else int(new_value)
                    if field in ['reps', 'sets'] and new_value <= 0:
                        await update.message.reply_text("Please enter a positive number.")
                        return
                except ValueError:
                    await update.message.reply_text(
                        f"❌ Invalid value. Please enter a {'number' if field == 'weight' else 'whole number'} for {field}."
                    )
                    return

            # Update the workout
            if self.update_workout(workout_id, field, new_value):
                # Get updated workout data
                workout = self.get_workout_by_id(workout_id)
                if not workout:
                    await update.message.reply_text("❌ Workout not found.")
                    return

                # Show the updated workout without success header
                message = self.format_workout_message(workout, include_header=False)
                message += "\nSelect what else you want to edit:"

                # Create edit menu again
                keyboard = [
                    [InlineKeyboardButton("💪 Muscle Group", callback_data=f"edit_field_muscle_group_{workout_id}")],
                    [InlineKeyboardButton("🎯 Exercise", callback_data=f"edit_field_exercise_{workout_id}")],
                    [InlineKeyboardButton("⚖️ Weight", callback_data=f"edit_field_weight_{workout_id}")],
                    [InlineKeyboardButton("🔄 Reps", callback_data=f"edit_field_reps_{workout_id}")],
                    [InlineKeyboardButton("📊 Sets", callback_data=f"edit_field_sets_{workout_id}")],
                    [InlineKeyboardButton("📝 Notes", callback_data=f"edit_field_notes_{workout_id}")],
                    [InlineKeyboardButton("✅ Done", callback_data="cancel_edit")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(message, reply_markup=reply_markup)
            else:
                await update.message.reply_text("❌ Failed to update workout")

            # Clear editing state
            del context.user_data['editing']

        except Exception as e:
            logging.error(f"Error handling edit text: {str(e)}")
            await update.message.reply_text("❌ An error occurred while updating the workout.")

            
    async def handle_delete_callback(self, update, context):
        """Handle delete workout callback"""
        query = update.callback_query
        user_id = query.from_user.id
        
        if query.data == "delete_workout_latest":
            workout = self.get_latest_workout(user_id)
            if not workout:
                await query.edit_message_text("Workout not found.")
                return

            keyboard = [
                [
                    InlineKeyboardButton("✅ Yes", callback_data=f"confirm_delete_{workout['id']}"),
                    InlineKeyboardButton("❌ No", callback_data="cancel_delete")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "Are you sure you want to delete this workout?",
                reply_markup=reply_markup
            )
            return
            
    def delete_all_workouts(self, user_id):
        """Delete all workouts for a specific user"""
        try:
            result = self.db.workouts.delete_many({"user_id": user_id})
            return result.deleted_count
        except Exception as e:
            logging.error(f"Error deleting all workouts: {str(e)}")
            return 0
        
    async def erase(self, update, context):
        """Handle /erase command - Ask for confirmation before deleting all workouts"""
        try:
            keyboard = [
                [
                    InlineKeyboardButton("✅ Yes, delete everything", callback_data="confirm_erase_all"),
                    InlineKeyboardButton("❌ No, keep my data", callback_data="cancel_erase")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                "⚠️ Warning! This will delete ALL your workout history!\n\n"
                "Are you absolutely sure you want to erase everything?\n"
                "This action cannot be undone.",
                reply_markup=reply_markup
            )
        except Exception as e:
            logging.error(f"Error in erase command: {str(e)}")
            await update.message.reply_text("❌ An error occurred. Please try again.")

    async def handle_erase_callback(self, update, context):
        """Handle erase confirmation callbacks"""
        try:
            query = update.callback_query
            user_id = query.from_user.id
            
            await query.answer()  # Acknowledge the button press
            
            if query.data == "confirm_erase_all":
                # Delete all workouts
                deleted_count = self.delete_all_workouts(user_id)
                if deleted_count > 0:
                    await query.edit_message_text(
                        f"✅ Success! Deleted {deleted_count} workouts from your history.\n\n"
                        "You can start fresh by sending a new voice message with your workout!"
                    )
                else:
                    await query.edit_message_text("No workout history found to delete.")
            
            elif query.data == "cancel_erase":
                await query.edit_message_text("✅ Deletion cancelled. Your workout history is safe!")
                
        except Exception as e:
            logging.error(f"Error in erase callback: {str(e)}")
            await query.edit_message_text("❌ An error occurred. Please try again.")

    def parse_workout_info(self, transcription):
        """Parse transcription using GPT-4o mini"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.OPENAI_API_KEY}"
        }
        
        system_prompt = """
        Extract workout information from the transcribed text. Return only raw JSON with these fields:
        - muscle_group: The main muscle group worked
        - exercise: The specific exercise performed
        - weight: Weight used in kilograms (if mentioned in pounds, convert to kg by dividing by 2.20462)
        - reps: Number of repetitions (if mentioned)
        - sets: Number of sets (if mentioned)
        - notes: Any additional information about form, difficulty, or other observations
        
        If a field is not mentioned in the text, set it to null.
        If weight is mentioned in pounds, convert it to kilograms and round to 1 decimal place.
        Do not include markdown formatting or code blocks in the response.
        """
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {
                                "type": "text",
                                "text": transcription
                            }
                        ]}
                    ]
                }
            )
            
            if response.status_code == 200:
                try:
                    content = response.json()['choices'][0]['message']['content']
                    # Add logging to debug the content
                    logging.info(f"GPT-4o mini response content: {content}")
                    
                    # Clean up the content by removing markdown code blocks if present
                    cleaned_content = content.replace('```json', '').replace('```', '').strip()
                    
                    # Ensure the content is valid JSON
                    workout_info = json.loads(cleaned_content)
                    
                    if workout_info.get('weight'):
                        weight_str = str(workout_info['weight']).lower().replace('kg', '').strip()
                        workout_info['weight'] = round(float(weight_str), 1)
                    
                    return workout_info
                except json.JSONDecodeError as e:
                    logging.error(f"JSON parsing error: {str(e)}\nContent received: {content}")
                    return None
                except Exception as e:
                    logging.error(f"Error processing GPT response: {str(e)}")
                    return None
            else:
                logging.error(f"GPT-4o mini API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logging.error(f"Parsing error: {str(e)}\n{traceback.format_exc()}")
            return None
    
    async def run(self):
        """Start the bot"""
        application = ApplicationBuilder().token(self.TELEGRAM_TOKEN).build()
        
        # Command handlers
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help))
        application.add_handler(CommandHandler("history", self.history))
        application.add_handler(CommandHandler("export", self.export_workouts))
        application.add_handler(CommandHandler("erase", self.erase))
        application.add_handler(CommandHandler("stats", self.stats))
        application.add_handler(CommandHandler("merge", self.merge))  # Add merge command
        
        # Callback query handlers in specific order
        application.add_handler(CallbackQueryHandler(self.handle_edit_callback, pattern="^edit_workout_"))
        application.add_handler(CallbackQueryHandler(self.handle_delete_callback, pattern="^delete_workout_"))
        application.add_handler(CallbackQueryHandler(self.handle_erase_callback, pattern="^(confirm|cancel)_erase"))
        application.add_handler(CallbackQueryHandler(self.handle_merge_callback, pattern="^merge_"))  # Add merge callback
        application.add_handler(CallbackQueryHandler(self.handle_history_callback))
        
        # Message handlers
        application.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        
        # Text message handler (for editing)
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_edit_text
        ))
        
        logging.info("Starting bot...")
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        
        # Keep the application running
        stop_signal = asyncio.Event()
        await stop_signal.wait()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bot.log')
        ]
    )
    
    bot = None
    try:
        bot = WorkoutBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
    finally:
        if bot and hasattr(bot, 'mongo_client'):
            bot.mongo_client.close()