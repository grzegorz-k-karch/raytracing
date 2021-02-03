#ifndef LOGGING_H
#define LOGGING_H

#include <boost/log/trivial.hpp>

void initLogger(const boost::log::trivial::severity_level logLevel);

#endif//LOGGING_H
