#pragma once
#include <QString>
#include <Singleton.h>
class SuikaGraphics;

class Debug
{
public:
	friend class Singleton<Debug>;

	static void Log(QString info);
	static void LogError(QString info);
	static void LogSystem(QString info);

	void SetSuikaGraphics(SuikaGraphics* suikaGraphics)
	{
		m_pSuikaGraphics = suikaGraphics;
	}

	void m_Log(QString string);
	void m_LogError(QString string);
	void m_LogSystem(QString info);

private:
	Debug() {};
	SuikaGraphics* m_pSuikaGraphics;
};
