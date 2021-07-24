#include <Precompiled.h>
#include <SuikaGraphics.h>

void Debug::m_Log(QString info)
{
	QTime current_time = QTime::currentTime();
	m_pSuikaGraphics->AppendDebugInfo("<font color=\"#FFFFFF\">" + current_time.toString() + ": " + info + "</font>");
}

void Debug::m_LogError(QString info)
{
	QTime current_time = QTime::currentTime();
	m_pSuikaGraphics->AppendDebugInfo("<font color=\"#FF0000\">" + current_time.toString() + ": " + info + "</font>");
}

void Debug::m_LogSystem(QString info)
{
	QTime current_time = QTime::currentTime();
	m_pSuikaGraphics->AppendDebugInfo("<font color=\"#1122FF\">" + current_time.toString() + ": " + info + "</font>");
}

void Debug::Log(QString info)
{
	Debug& debug = Singleton<Debug>::get_instance();
	debug.m_Log(info);
}

void Debug::LogError(QString info)
{
	Debug& debug = Singleton<Debug>::get_instance();
	debug.m_LogError(info);
}

void Debug::LogSystem(QString info)
{
	Debug& debug = Singleton<Debug>::get_instance();
	debug.m_LogSystem(info);
}