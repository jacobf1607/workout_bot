# 🏋️‍♂️ Telegram Workout Tracking Bot

A powerful Telegram bot for tracking workouts through voice messages, with features for progress visualization, statistics, and workout history management.

## Features

### Core Functionality
- 🎤 Voice message workout logging with natural language processing
- 💪 Automatic exercise and muscle group categorization
- ⚖️ Automatic pounds to kilograms conversion
- 📊 Progress tracking and visualization
- 📈 Personal records and statistics

### Data Management
- 📥 Export workout history to Excel
- ✏️ Edit or delete logged workouts
- 🔄 Merge similar exercise names
- 🗑️ Full workout history cleanup option

### Progress Tracking
- 📊 Visual progress graphs for each exercise
- 📅 Workout history organized by muscle groups
- 🏆 Personal records tracking
- 📈 Performance statistics and trends

## Tech Stack

- **Python 3.8+**
- **MongoDB** for data storage
- **OpenAI API** for voice transcription and workout parsing
- **Telegram Bot API**
- **Matplotlib** for data visualization
- **Pandas** for data processing

## Prerequisites

1. Python 3.8 or higher
2. MongoDB instance
3. Telegram Bot Token
4. OpenAI API Key

## Environment Variables

Create a `.env` file with the following variables:

```env
MONGODB_URI=your_mongodb_connection_string
TELEGRAM_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jacobf1607/workout_bot.git
cd telegram-workout-bot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env` file

4. Run the bot:
```bash
python workout_bot.py
```

## Usage

1. Start a chat with the bot on Telegram
2. Send `/start` to see available commands
3. Record a voice message describing your workout:
   - Example: "Chest workout, bench press, 100kgs, 3 sets of 8 reps"

### Available Commands

- `/start` - Initialize the bot and see usage instructions
- `/history` - View workout history and progress graphs
- `/stats` - See workout statistics and achievements
- `/export` - Download workout history as Excel file
- `/merge` - Combine duplicate exercise names
- `/erase` - Delete all workout history
- `/help` - Show detailed usage information

## Features in Detail

### Voice Message Processing
The bot uses OpenAI's Whisper API for voice transcription and GPT for parsing workout details. It automatically extracts:
- Muscle group
- Exercise name
- Weight (with automatic lb to kg conversion)
- Sets and reps
- Additional notes

### Progress Tracking
- Visual graphs showing progress over time
- Weight, reps, and sets tracked separately
- Personal records for each exercise
- Comprehensive statistics

### Data Management
- Export complete workout history to Excel
- Edit workout details after logging
- Merge similar exercise names to maintain clean data
- Option to reset all data if needed

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

MIT License - see LICENSE file for details
