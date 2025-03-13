#pragma once

#include <chrono>
#include <mutex>
#include <sstream>
#include <string>

enum class LogLevel {
	OFF      = 0,
	CRITICAL = 1,
	ERR      = 2,
	WARN     = 3,
	INFO     = 4,
	DEBUG    = 5,
	TRACE    = 6
};

struct Logger {

public:
	// Constructor
	Logger(LogLevel logLevel);

	// Log function printing messages based on their level
	void Log(LogLevel messageLevel, const char* message);

	// Overloading functions
	void Log(LogLevel messageLevel, const std::string& message) {
		Log(messageLevel, message.c_str());
	}

	void Log(LogLevel messageLevel, const std::ostringstream& streamOutput) {
		Log(messageLevel, streamOutput.str());
	}

	// Used to display the progress of the current status
	void displayProgressBar(const size_t currentIteration,
		const size_t totalIterations);

	LogLevel level;

private:
	// Used to compute time between every logging event
	std::chrono::system_clock::time_point currentTime;
	std::chrono::system_clock::time_point lastLogTime;
	std::chrono::milliseconds elapsedTime;

	// Used in order to print current and elapsed time
	time_t currentTimeInstanced; char currentTimeChar[26];
	std::string currentTimeString, elapsedTimeString;

	// Used to update a progress bar for current status
	std::atomic<size_t> currentIteration{ 0 };
	std::atomic<size_t> numberIterations{ 1 };

	std::mutex logMutex; // mutex added for thread-safety

};

// Creating macros for using the logger
#define LOG(level, msg)   logger.Log(level, std::string(__FUNCTION__) + ": " + msg)
#define LOG_TRACE(msg)    logger.Log(LogLevel::TRACE   , std::string(__FUNCTION__) + ": " + msg)
#define LOG_DEBUG(msg)    logger.Log(LogLevel::DEBUG   , std::string(__FUNCTION__) + ": " + msg)
#define LOG_INFO(msg)     logger.Log(LogLevel::INFO    , std::string(__FUNCTION__) + ": " + msg)
#define LOG_WARN(msg)     logger.Log(LogLevel::WARN    , std::string(__FUNCTION__) + ": " + msg)
#define LOG_ERR(msg)      logger.Log(LogLevel::ERR     , std::string(__FUNCTION__) + ": " + msg)
#define LOG_CRITICAL(msg) logger.Log(LogLevel::CRITICAL, std::string(__FUNCTION__) + ": " + msg)

extern Logger logger;

