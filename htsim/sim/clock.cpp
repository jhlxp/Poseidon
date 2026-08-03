// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-        
#include <iostream>
#include "clock.h"
#include "eventlist.h"

Clock::Clock(simtime_picosec period, EventList& eventlist)
  : EventSource(eventlist,"clock"), 
    _period(period), _smallticks(0)
{
    eventlist.sourceIsPendingRel(*this, period);
}

void
Clock::doNextEvent() {
    eventlist().sourceIsPendingRel(*this, _period);
}
