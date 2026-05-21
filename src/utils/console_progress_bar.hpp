#ifndef __RT_ISICG_CONSOLE_PROGRESS_BAR__
#define __RT_ISICG_CONSOLE_PROGRESS_BAR__

#include <mutex>

class ConsoleProgressBar {
  public:
    ConsoleProgressBar() = default;

    void start(const int p_nbTasks, const int p_barWidth);
    void next(const int n = 1);
    void stop();

  private:
    int _barWidth = 50;
    int _nbTasksDone = 0;
    int _nbTasks = 1;

    std::mutex _mutex;
};

#endif // __RT_ISICG_CONSOLE_PROGRESS_BAR__
