#pragma once

#include <optional>
#include <ryml.hpp>
#include <ryml_std.hpp>

  /*
   * @brief The following function is used to get the value of a parameter in a list (node) in
   * a yaml tree
   */
  template<class T>
  T rget(ryml::ConstNodeRef node, c4::csubstr key);

  /*
   * \brief The following function is for getting an optional parameter
   */
  template <class T>
  std::optional<T> rget_optional(ryml::ConstNodeRef node, c4::csubstr key);
  