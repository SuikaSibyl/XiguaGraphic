#include <Precompiled.h>

#include "linear_algebra.h"
#include "geometry.h"
#include <CudaPathTracer.h>

using std::string;

namespace enums {
	enum ColorComponent {
		Red = 0,
		Green = 1,
		Blue = 2
	};
}

using namespace enums;

// Rescale input objects to have this size...
const float MaxCoordAfterRescale = 1.2f;
const float Scale = 10;

// if some file cannot be found, panic and exit
void panic(const char* fmt, ...)
{
	static char message[131072];
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(message, sizeof message, fmt, ap);
	printf(message); fflush(stdout);
	va_end(ap);

	exit(1);
}

void fix_normals(void)
{

}

float load_object(const char* filename)
{

return 1;
}
