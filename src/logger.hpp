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

	template <typename... Args>
	void Log(LogLevel messagelevel, Args... args) {
		std::ostringstream streamOutput;
		(streamOutput << ... << args);
		Log(messagelevel, streamOutput.str());
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
#define LOG(level, ...)   logger.Log(level, std::string(__FUNCTION__) + ": ", __VA_ARGS__)
#define LOG_TRACE(...)    logger.Log(LogLevel::TRACE   , std::string(__FUNCTION__) + ": ", __VA_ARGS__)
#define LOG_DEBUG(...)    logger.Log(LogLevel::DEBUG   , std::string(__FUNCTION__) + ": ", __VA_ARGS__)
#define LOG_INFO(...)     logger.Log(LogLevel::INFO    , std::string(__FUNCTION__) + ": ", __VA_ARGS__)
#define LOG_WARN(...)     logger.Log(LogLevel::WARN    , std::string(__FUNCTION__) + ": ", __VA_ARGS__)
#define LOG_ERR(...)      logger.Log(LogLevel::ERR     , std::string(__FUNCTION__) + ": ", __VA_ARGS__)
#define LOG_CRITICAL(...) logger.Log(LogLevel::CRITICAL, std::string(__FUNCTION__) + ": ", __VA_ARGS__)

extern Logger logger;

