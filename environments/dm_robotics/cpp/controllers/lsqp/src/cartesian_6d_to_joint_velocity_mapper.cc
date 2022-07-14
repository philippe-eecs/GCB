// Copyright 2020 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dm_robotics/controllers/lsqp/cartesian_6d_to_joint_velocity_mapper.h"

#include <utility>

#include "dm_robotics/support/logging.h"
#include "absl/container/btree_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "dm_robotics/controllers/lsqp/cartesian_6d_velocity_task.h"
#include "dm_robotics/controllers/lsqp/collision_avoidance_constraint.h"
#include "dm_robotics/controllers/lsqp/joint_acceleration_constraint.h"
#include "dm_robotics/controllers/lsqp/joint_position_limit_constraint.h"
#include "dm_robotics/least_squares_qp/common/box_constraint.h"
#include "dm_robotics/least_squares_qp/common/identity_constraint_union.h"
#include "dm_robotics/least_squares_qp/common/identity_task.h"
#include "dm_robotics/least_squares_qp/common/minimize_norm_task.h"
#include "dm_robotics/least_squares_qp/core/lsqp_stack_of_tasks_solver.h"
#include "dm_robotics/mujoco/mjlib.h"
#include "dm_robotics/mujoco/types.h"
#include "dm_robotics/mujoco/utils.h"
#include "dm_robotics/support/status_macros.h"

namespace dm_robotics {
namespace {

// Constructs an LsqpStackOfTasksSolver::Parameters object from
// a Cartesian6dToJointVelocityMapper::Parameters object.
LsqpStackOfTasksSolver::Parameters ToLsqpParams(
    const Cartesian6dToJointVelocityMapper::Parameters& params) {
  LsqpStackOfTasksSolver::Parameters output_params;
  output_params.absolute_tolerance = params.solution_tolerance;
  output_params.relative_tolerance = 0.0;
  output_params.hierarchical_projection_slack =
      params.nullspace_projection_slack;

  // We always ensure that the infeasibility tolerance is less than the absolute
  // tolerance, by making them a tenth of the absolute tolerance. This prevents
  // the optimizer from thinking a solution is infeasible too easily.
  output_params.primal_infeasibility_tolerance =
      0.1 * params.solution_tolerance;
  output_params.dual_infeasibility_tolerance = 0.1 * params.solution_tolerance;

  // If nullspace failures do not return an error, add a warning when it fails
  // if `log_nullspace_failure_warnings` is `true`.
  output_params.return_error_on_nullspace_failure =
      params.return_error_on_nullspace_failure;
  if (!params.return_error_on_nullspace_failure &&
      params.log_nullspace_failure_warnings) {
    output_params.verbosity =
        LsqpStackOfTasksSolver::Parameters::VerboseFlags::kNullspaceWarnings;
  } else {
    output_params.verbosity =
        LsqpStackOfTasksSolver::Parameters::VerboseFlags::kNone;
  }
  return output_params;
}

// Constructs a Cartesian6dVelocityTaskParameters object from
// a Cartesian6dToJointVelocityMapper::Parameters object.
Cartesian6dVelocityTask::Parameters ToCartesianVelocityParams(
    const Cartesian6dToJointVelocityMapper::Parameters& params) {
  Cartesian6dVelocityTask::Parameters output_params;
  output_params.lib = params.lib;
  output_params.model = params.model;
  output_params.joint_ids = params.joint_ids;
  output_params.object_type = params.object_type;
  output_params.object_name = params.object_name;
  output_params.weighting_matrix =
      params.cartesian_velocity_task_weighting_matrix;
  return output_params;
}

// Constructs a CollisionAvoidanceConstraintParameters object from
// a Cartesian6dToJointVelocityMapper::Parameters object.
CollisionAvoidanceConstraint::Parameters ToCollisionAvoidanceParams(
    const Cartesian6dToJointVelocityMapper::Parameters& params) {
  CollisionAvoidanceConstraint::Parameters output_params;
  output_params.lib = params.lib;
  output_params.model = params.model;
  output_params.collision_detection_distance =
      params.collision_detection_distance;
  output_params.minimum_normal_distance =
      params.minimum_distance_from_collisions;
  output_params.gain = params.collision_avoidance_normal_velocity_scale /
                       absl::ToDoubleSeconds(params.integration_timestep);

  // We add a negative offset on the upper bound of each collision avoidance
  // constraint to prevent violations due to numerical errors. This is because
  // OSQP constraint violations depend on the solution tolerance, and thus the
  // computed velocities may result in the normal velocity penetrating the
  // minimum allowable distance.
  // Although this effectively removes the zero-vector as an always valid
  // solution to the optimization problem, we found in practice that it prevents
  // collisions due to compounded numerical integration errors without
  // noticeably making the problem more difficult to solve.
  output_params.bound_relaxation = -params.solution_tolerance;

  output_params.joint_ids = params.joint_ids;

  // Similar to MuJoCo, we do not allow parent-child collisions. However, unlike
  // MuJoCo, we only allow collisions with the worldbody geoms if the pair is
  // explicitly specified. This is because in robot environments, the first body
  // of the kinematic chain may be welded to the worldbody, and MuJoCo's default
  // behaviour would result in collisions between this first body and its child,
  // even though they have a parent-child relationship. This is not necessary
  // for most robot environments, as joint limits are such that parent-children
  // cannot collide.
  output_params.geom_pairs = CollisionPairsToGeomIdPairs(
      *params.lib, *params.model, params.collision_pairs, false, false);
  return output_params;
}

// Constructs a JointPositionLimitConstraintParameters object from
// a Cartesian6dToJointVelocityMapper::Parameters object.
JointPositionLimitConstraint::Parameters ToJointPositionLimitParams(
    const Cartesian6dToJointVelocityMapper::Parameters& params) {
  JointPositionLimitConstraint::Parameters output_params;
  output_params.model = params.model;
  output_params.min_distance_from_limits =
      params.minimum_distance_from_joint_position_limit;
  output_params.gain = params.joint_position_limit_velocity_scale /
                       absl::ToDoubleSeconds(params.integration_timestep);
  output_params.joint_ids = params.joint_ids;
  return output_params;
}

// Constructs a JointAccelerationConstraintParameters object from
// a Cartesian6dToJointVelocityMapper::Parameters object.
JointAccelerationConstraint::Parameters ToJointAccelerationLimitParams(
    const Cartesian6dToJointVelocityMapper::Parameters& params) {
  JointAccelerationConstraint::Parameters output_params;
  output_params.model = params.model;
  output_params.joint_ids = params.joint_ids;
  output_params.integration_timestep = params.integration_timestep;
  output_params.acceleration_magnitude_limits =
      params.joint_acceleration_magnitude_limits;
  return output_params;
}

// Clips each element in `values` to be within the respective elements in
// `lower_bound` and `upper_bound`. All arrays are assumed to be the same size.
void ClipToBounds(absl::Span<const double> lower_bound,
                  absl::Span<const double> upper_bound,
                  absl::Span<double> values) {
  for (int i = 0; i < values.size(); ++i) {
    values[i] = std::clamp(values[i], lower_bound[i], upper_bound[i]);
  }
}

// For each pair of geoms, computes the minimum distance between both geoms by
// using MuJoCo's collision detection mechanism with
// `collision_detection_distance` as the margin. If several contacts are
// detected, only the smallest distance is saved. If no contacts are detected,
// no value will be set for that pair.
void ComputeGeomPairDist(
    const MjLib& lib, const mjModel& model, const mjData& data,
    double collision_detection_distance,
    absl::btree_map<std::pair<int, int>, absl::optional<double>>*
        geom_pair_dist) {
  for (auto& [pair, maybe_dist] : *geom_pair_dist) {
    maybe_dist =
        ComputeMinimumContactDistance(lib, model, data, pair.first, pair.second,
                                      collision_detection_distance);
  }
}

// Integrates MuJoCo positions with an array of `joint_velocities`, ordered
// according to the `joint_dof_ids` object.
void IntegrateQpos(const MjLib& lib, const mjModel& model,
                   absl::Duration integration_timestep,
                   const absl::btree_set<int>& joint_dof_ids,
                   absl::Span<const double> joint_velocities, mjData* data) {
  int joint_velocity_idx = 0;
  for (int dof_id : joint_dof_ids) {
    data->qvel[dof_id] = joint_velocities[joint_velocity_idx];
    ++joint_velocity_idx;
  }
  lib.mj_integratePos(&model, data->qpos, data->qvel,
                      absl::ToDoubleSeconds(integration_timestep));
}

// Returns true if penetration is increased for any geom pair, false otherwise.
// Penetration is increased if the objects are penetrating and the distance
// becomes more negative, or if the objects were not penetrating and start
// panetrating after. If this function returns true, penetrating_pair is filled
// with the offending geoms IDs.
//
// Assumes both maps have exactly the same keys.
bool IsPenetrationIncreased(
    double minimum_distance_from_collisions,
    const absl::btree_map<std::pair<int, int>, absl::optional<double>>&
        geom_pair_dist_before,
    const absl::btree_map<std::pair<int, int>, absl::optional<double>>&
        geom_pair_dist_after,
    std::pair<int, int>* penetrating_pair) {
  auto before = geom_pair_dist_before.begin();
  auto after = geom_pair_dist_after.begin();
  while (before != geom_pair_dist_before.end()) {
    const auto& [pair_after, maybe_dist_after] = *after;
    const auto& [pair_before, maybe_dist_before] = *before;
    bool has_penetration_after =
        maybe_dist_after.has_value() &&
        maybe_dist_after.value() < minimum_distance_from_collisions;
    bool penetration_increased =
        has_penetration_after &&
        (!maybe_dist_before.has_value() ||
         maybe_dist_before.value() > maybe_dist_after.value());
    if (penetration_increased) {
      *penetrating_pair = pair_after;
      return true;
    }
    ++before;
    ++after;
  }
  return false;
}

// Returns an OK status if all the geoms in `group` are valid geoms in `model`.
absl::Status ValidateGeomGroup(const MjLib& lib, const mjModel& model,
                               const GeomGroup& group) {
  for (const auto& name : group) {
    bool is_name_found =
        lib.mj_name2id(&model, mjtObj::mjOBJ_GEOM, name.c_str()) >= 0;
    if (!is_name_found) {
      return absl::NotFoundError(
          absl::Substitute("ValidateGeomGroup: Could not find MuJoCo geom with "
                           "name [$0] in the provided model.",
                           name));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status Cartesian6dToJointVelocityMapper::ValidateParameters(
    const Parameters& params) {
  if (params.lib == nullptr) {
    return absl::InvalidArgumentError(
        "ValidateParameters: `lib` cannot be null.");
  }
  if (params.model == nullptr) {
    return absl::InvalidArgumentError(
        "ValidateParameters: `model` cannot be null.");
  }
  if (params.joint_ids.empty()) {
    return absl::InvalidArgumentError(
        "ValidateParameters: `joint_ids` cannot be empty.");
  }

  for (int joint_id : params.joint_ids) {
    if (joint_id < 0 || joint_id >= params.model->njnt)
      return absl::OutOfRangeError(absl::Substitute(
          "ValidateParameters: Provided joint_id [$0] is invalid for the "
          "provided model, which has [$1] joints.",
          joint_id, params.model->njnt));
  }

  int number_of_dof = JointIdsToDofIds(*params.model, params.joint_ids).size();
  if (number_of_dof != params.joint_ids.size()) {
    return absl::InvalidArgumentError(absl::Substitute(
        "ValidateParameters: `joint_ids` must only contain 1 DoF joints. "
        "Number of joints [$0]; number of DoF [$1].",
        params.joint_ids.size(), number_of_dof));
  }

  bool is_object_type_valid = params.object_type == mjtObj::mjOBJ_BODY ||
                              params.object_type == mjtObj::mjOBJ_GEOM ||
                              params.object_type == mjtObj::mjOBJ_SITE;
  if (!is_object_type_valid)
    return absl::InvalidArgumentError(absl::Substitute(
        "ValidateParameters: Objects of type [$0] are not supported. Only "
        "bodies, geoms, and sites are supported.",
        params.lib->mju_type2Str(params.object_type)));

  bool is_object_found =
      params.lib->mj_name2id(params.model, params.object_type,
                             params.object_name.c_str()) >= 0;
  if (!is_object_found)
    return absl::NotFoundError(absl::Substitute(
        "ValidateParameters: Could not find MuJoCo object with name [$0] and "
        "type [$1] in the provided model.",
        params.object_name, params.lib->mju_type2Str(params.object_type)));

  if (params.integration_timestep <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError(
        absl::Substitute("ValidateParameters: `integration_timestep` [$0 s] "
                         "must be a positive duration.",
                         absl::ToDoubleSeconds(params.integration_timestep)));
  }

  if (params.enable_joint_velocity_limits) {
    if (params.joint_velocity_magnitude_limits.size() != number_of_dof) {
      return absl::InvalidArgumentError(absl::Substitute(
          "ValidateParameters: joint velocity limits are enabled but size of "
          "`joint_velocity_magnitude_limits` [$0] does not match the number of "
          "DoF [$1].",
          params.joint_velocity_magnitude_limits.size(), number_of_dof));
    }
  }

  if (params.enable_joint_acceleration_limits) {
    if (params.joint_acceleration_magnitude_limits.size() != number_of_dof) {
      return absl::InvalidArgumentError(absl::Substitute(
          "ValidateParameters: joint acceleration limits are enabled but size "
          "of `joint_acceleration_magnitude_limits` [$0] does not match the "
          "number of DoF [$1].",
          params.joint_acceleration_magnitude_limits.size(), number_of_dof));
    }
  }

  for (const auto& collision_pair : params.collision_pairs) {
    RETURN_IF_ERROR(
        ValidateGeomGroup(*params.lib, *params.model, collision_pair.first));
    RETURN_IF_ERROR(
        ValidateGeomGroup(*params.lib, *params.model, collision_pair.second));
  }

  if (params.max_cartesian_velocity_control_iterations <= 0) {
    return absl::InvalidArgumentError(absl::Substitute(
        "ValidateParameters: `max_cartesian_velocity_control_iterations` [$0] "
        "must be positive.",
        params.max_cartesian_velocity_control_iterations));
  }

  if (params.solution_tolerance <= 0.0) {
    return absl::InvalidArgumentError(absl::Substitute(
        "ValidateParameters: `solution_tolerance` [$0] must be positive.",
        params.solution_tolerance));
  }

  if (params.enable_nullspace_control) {
    if (params.max_nullspace_control_iterations <= 0) {
      return absl::InvalidArgumentError(absl::Substitute(
          "ValidateParameters: `max_nullspace_control_iterations` [$0] must be "
          "positive if nullspace control is enabled.",
          params.max_nullspace_control_iterations));
    }
    if (params.nullspace_projection_slack <= 0) {
      return absl::InvalidArgumentError(absl::Substitute(
          "ValidateParameters: `nullspace_projection_slack` [$0] must be "
          "positive if nullspace control is enabled.",
          params.nullspace_projection_slack));
    }
  }
  return absl::OkStatus();
}

Cartesian6dToJointVelocityMapper::Cartesian6dToJointVelocityMapper(
    const Parameters& params)
    : lib_(*DieIfNull(params.lib)),
      model_(*DieIfNull(params.model)),
      data_(lib_.mj_makeData(&model_), lib_.mj_deleteData),
      joint_dof_ids_(JointIdsToDofIds(model_, params.joint_ids)),
      integration_timestep_(params.integration_timestep),
      clamp_nullspace_bias_to_feasible_space_(
          params.clamp_nullspace_bias_to_feasible_space),
      remove_joint_acceleration_limits_if_in_conflict_(
          params.remove_joint_acceleration_limits_if_in_conflict),
      minimum_distance_from_collisions_(
          params.minimum_distance_from_collisions),
      qp_solver_(ToLsqpParams(params)),
      solution_(joint_dof_ids_.size()),
      check_solution_validity_(params.check_solution_validity),
      log_collision_warnings_(params.log_collision_warnings &&
                              params.enable_collision_avoidance),
      cartesian_velocity_task_(nullptr),
      nullspace_task_(nullptr),
      joint_kinematic_constraints_(nullptr),
      collision_avoidance_constraint_(nullptr) {
  CHECK(params.integration_timestep > absl::ZeroDuration())
      << "Cartesian6dToJointVelocityMapper: `integration_timestep` ["
      << params.integration_timestep << "] must be a positive duration.";

  // Ensure data is in a valid state.
  std::fill_n(data_->qvel, model_.nv, 0.0);
  lib_.mj_fwdPosition(&model_, data_.get());

  // First hierarchy has the Cartesian velocity task and the regularization
  // task for singularity robustness. The weight for the Cartesian velocity task
  // is always 1.0, and the regularization weight is defined relative to this.
  auto first_hierarchy = qp_solver_.AddNewTaskHierarchy(
      params.max_cartesian_velocity_control_iterations);
  cartesian_velocity_task_ =
      first_hierarchy
          ->InsertOrAssignTask("CartesianVelocity",
                               absl::make_unique<Cartesian6dVelocityTask>(
                                   ToCartesianVelocityParams(params), *data_,
                                   std::array<double, 6>()),
                               1.0, false)
          .first;
  first_hierarchy->InsertOrAssignTask(
      "Regularization",
      absl::make_unique<MinimizeNormTask>(joint_dof_ids_.size()),
      params.regularization_weight, true);

  // Second hierarchy has the nullspace task. This is created only if the user
  // enables nullspace control. The weight for the nullspace task is always 1.0,
  // as it is the only task.
  if (params.enable_nullspace_control) {
    nullspace_task_ =
        qp_solver_.AddNewTaskHierarchy(params.max_nullspace_control_iterations)
            ->InsertOrAssignTask(
                "NullspaceBias",
                absl::make_unique<IdentityTask>(
                    std::vector<double>(joint_dof_ids_.size(), 0.0)),
                1.0, true)
            .first;
    clipped_nullspace_bias_.resize(joint_dof_ids_.size());
  }

  // The LSQP solver only owns two constraints. A union of identity constraints
  // (which will combine all joint constraints into one), and the collision
  // avoidance constraint.
  // Add collision avoidance constraint if enabled.
  if (params.enable_collision_avoidance) {
    CollisionAvoidanceConstraint::Parameters collision_params =
        ToCollisionAvoidanceParams(params);
    collision_avoidance_constraint_ =
        qp_solver_
            .InsertOrAssignConstraint(
                "CollisionAvoidance",
                absl::make_unique<CollisionAvoidanceConstraint>(
                    collision_params, *data_))
            .first;

    // We use this to sanity-check the solution and ensure it is never in
    // collision. This is because MuJoCo's computed normals are sometimes
    // inaccurate, and the computed velocity may thus result in collisions when
    // integrated.
    for (auto geom_pair : collision_params.geom_pairs) {
      geom_pair_to_dist_curr_[geom_pair] = absl::nullopt;
      geom_pair_to_dist_after_[geom_pair] = absl::nullopt;
    }
  }

  // Create the joint position limit constraint if enabled, and push it into the
  // enabled identity constraints vector.
  if (params.enable_joint_position_limits) {
    joint_position_limit_constraint_ =
        absl::make_unique<JointPositionLimitConstraint>(
            ToJointPositionLimitParams(params), *data_);
    enabled_joint_kinematic_constraints_.push_back(
        joint_position_limit_constraint_.get());
  }

  // Create the joint velocity limit constraint if enabled, and push it into the
  // enabled identity constraints vector.
  if (params.enable_joint_velocity_limits) {
    std::vector<double> lower_bound(joint_dof_ids_.size());
    std::vector<double> upper_bound(joint_dof_ids_.size());
    for (int i = 0; i < params.joint_velocity_magnitude_limits.size(); ++i) {
      lower_bound[i] = -std::abs(params.joint_velocity_magnitude_limits[i]);
      upper_bound[i] = std::abs(params.joint_velocity_magnitude_limits[i]);
    }

    joint_velocity_limit_constraint_ =
        absl::make_unique<BoxConstraint>(lower_bound, upper_bound);
    enabled_joint_kinematic_constraints_.push_back(
        joint_velocity_limit_constraint_.get());
  }

  // Create the joint acceleration limit constraint if enabled, and push it into
  // the enabled identity constraints vector.
  if (params.enable_joint_acceleration_limits) {
    joint_acceleration_limit_constraint_ =
        absl::make_unique<JointAccelerationConstraint>(
            ToJointAccelerationLimitParams(params), *data_);
    enabled_joint_kinematic_constraints_.push_back(
        joint_acceleration_limit_constraint_.get());
  }

  // Create a union for merging the joint kinematic constraints, if at least one
  // is enabled.
  if (!enabled_joint_kinematic_constraints_.empty()) {
    joint_kinematic_constraints_ =
        qp_solver_
            .InsertOrAssignConstraint(
                "JointLimitConstraints",
                absl::make_unique<IdentityConstraintUnion>(
                    joint_dof_ids_.size()))
            .first;
  }

  // Resize solution buffer and setup problem.
  CHECK_EQ(qp_solver_.SetupProblem(), absl::OkStatus())
      << "Cartesian6dToJointVelocityMapper: Unable to setup problem. This "
         "should never happen. Please contact the developers for more "
         "information.";
}

absl::StatusOr<absl::Span<const double>>
Cartesian6dToJointVelocityMapper::ComputeJointVelocities(
    const mjData& data, absl::Span<const double> target_6d_cartesian_velocity,
    absl::Span<const double> nullspace_bias) {
  CHECK(nullspace_task_ != nullptr)
      << "Cartesian6dToJointVelocityMapper::ComputeJointVelocities: Function "
         "overload with `nullspace_bias` parameter must not be called when "
         "`enable_nullspace_control` is false.";
  CHECK(nullspace_bias.size() == joint_dof_ids_.size())
      << "Cartesian6dToJointVelocityMapper::ComputeJointVelocities: Size "
         "of the `nullspace_bias` array ["
      << nullspace_bias.size()
      << "] does not match the number of DoF being controlled ["
      << joint_dof_ids_.size() << "].";
  return ComputeJointVelocitiesImpl(data, target_6d_cartesian_velocity,
                                    nullspace_bias);
}

absl::StatusOr<absl::Span<const double>>
Cartesian6dToJointVelocityMapper::ComputeJointVelocities(
    const mjData& data, absl::Span<const double> target_6d_cartesian_velocity) {
  CHECK(nullspace_task_ == nullptr)
      << "Cartesian6dToJointVelocityMapper::ComputeJointVelocities: Function "
         "overload without `nullspace_bias` parameter must not be called when "
         "`enable_nullspace_control` is true.";
  return ComputeJointVelocitiesImpl(data, target_6d_cartesian_velocity,
                                    absl::Span<const double>());
}

absl::StatusOr<absl::Span<const double>>
Cartesian6dToJointVelocityMapper::ComputeJointVelocitiesImpl(
    const mjData& data, absl::Span<const double> target_6d_cartesian_velocity,
    absl::Span<const double> nullspace_bias) {
  // Update internal mjData and run necessary MuJoCo routines.
  std::copy_n(data.qpos, model_.nq, data_->qpos);
  std::copy_n(data.qvel, model_.nv, data_->qvel);
  lib_.mj_kinematics(&model_, data_.get());
  lib_.mj_comPos(&model_, data_.get());

  // We update the constraints first, as we need to clip the nullspace bias to
  // the resulting kinematic constraint bounds.
  RETURN_IF_ERROR(UpdateConstraints());
  RETURN_IF_ERROR(UpdateTasks(target_6d_cartesian_velocity, nullspace_bias));

  // Solve, copy solution into internal buffer, and clip to kinematic
  // constraints to prevent constraint violations due to numerical errors.
  ASSIGN_OR_RETURN(absl::Span<const double> solution, qp_solver_.Solve());
  std::copy(solution.begin(), solution.end(), solution_.begin());
  if (!enabled_joint_kinematic_constraints_.empty()) {
    ClipToBounds(joint_kinematic_constraints_->GetLowerBound(),
                 joint_kinematic_constraints_->GetUpperBound(),
                 absl::MakeSpan(solution_));
  }

  // Check solution validity if necessary.
  if (check_solution_validity_) {
    RETURN_IF_ERROR(CheckSolutionValidity());
  }

  return solution_;
}

// Assumes all tasks have been initialized, and that `UpdateConstraints` has
// been called.
absl::Status Cartesian6dToJointVelocityMapper::UpdateTasks(
    absl::Span<const double> target_6d_cartesian_velocity,
    absl::Span<const double> nullspace_bias) {
  // Update Cartesian task.
  cartesian_velocity_task_->UpdateCoefficientsAndBias(
      *data_, target_6d_cartesian_velocity);

  // Update nullspace task if it exists, with the clipped bounds.
  if (nullspace_task_ != nullptr) {
    std::copy(nullspace_bias.begin(), nullspace_bias.end(),
              clipped_nullspace_bias_.begin());

    // Clip nullspace to merged kinematic bounds, if they exist. This makes the
    // nullspace problem easier.
    if (clamp_nullspace_bias_to_feasible_space_ &&
        !enabled_joint_kinematic_constraints_.empty()) {
      ClipToBounds(joint_kinematic_constraints_->GetLowerBound(),
                   joint_kinematic_constraints_->GetUpperBound(),
                   absl::MakeSpan(clipped_nullspace_bias_));
    }
    nullspace_task_->SetTarget(clipped_nullspace_bias_);
  }
  return absl::OkStatus();
}

// Assumes all constraints are initialized, and `data_` has been updated with
// the latest joint positions and joint velocities.
absl::Status Cartesian6dToJointVelocityMapper::UpdateConstraints() {
  bool has_joint_position_limit = joint_position_limit_constraint_ != nullptr;
  bool has_joint_acceleration_limit =
      joint_acceleration_limit_constraint_ != nullptr;
  bool has_joint_kinematic_constraint =
      !enabled_joint_kinematic_constraints_.empty();
  bool has_collision_avoidance_constraint =
      collision_avoidance_constraint_ != nullptr;

  if (has_joint_kinematic_constraint) {
    // Update each joint kinematic constraint, if they exist, and merge them
    // into a single identity constraint. Note that the joint velocity limit
    // constraint does not need to be updated, as it is constant.
    if (has_joint_position_limit) {
      joint_position_limit_constraint_->UpdateBounds(*data_);
    }
    if (has_joint_acceleration_limit) {
      joint_acceleration_limit_constraint_->UpdateBounds(*data_);
    }
    absl::Status kinematic_constraints_status =
        joint_kinematic_constraints_->UpdateFeasibleSpace(
            enabled_joint_kinematic_constraints_);

    // If joint acceleration limits are not enabled, updating the feasible space
    // must never fail. If it does, there is an internal programming bug
    // somewhere. Note that the joint position limits and joint velocity limits
    // constraint must never conflict, as the zero-vector will always be a
    // solution.
    if (!has_joint_acceleration_limit) {
      CHECK_EQ(kinematic_constraints_status, absl::OkStatus())
          << "Cartesian6dToJointVelocityMapper::UpdateConstraints: The "
             "joint position limit and joint velocity limit constraints "
             "are in conflict ["
          << kinematic_constraints_status.ToString()
          << "]. This should never happen; please "
             "contact the developers for more information.";
    }

    // The acceleration limit must be the cause of the confict iff:
    //   * The kinematic constraints are in conflict;
    //   * The acceleration limit exists;
    //   * The acceleration limit is not the only kinematic constraint.
    bool acceleration_limit_conflicts =
        !kinematic_constraints_status.ok() && has_joint_acceleration_limit &&
        enabled_joint_kinematic_constraints_.size() != 1;

    // If solver is not asked to remove the conflict, return an error instead.
    if (acceleration_limit_conflicts &&
        !remove_joint_acceleration_limits_if_in_conflict_) {
      return absl::NotFoundError(absl::Substitute(
          "Cartesian6dToJointVelocityMapper::UpdateConstraints: The joint "
          "acceleration limit constraint is in conflict with one or more joint "
          "limit constraints in this configuration. The resulting optimization "
          "problem is infeasible [$0]. To prevent this error, the user may "
          "want to increase the acceleration limits or set the "
          "`remove_joint_acceleration_limits_if_in_conflict` parameter to "
          "true.",
          kinematic_constraints_status.ToString()));
    } else if (acceleration_limit_conflicts &&
               remove_joint_acceleration_limits_if_in_conflict_) {
      // The acceleration limit constraint will always be the last constraint in
      // the array. If after removing the acceleration limit, the union still
      // fails to find a feasible space, then we have an internal programming
      // bug somewhere.
      CHECK_EQ(joint_kinematic_constraints_->UpdateFeasibleSpace(absl::MakeSpan(
                   enabled_joint_kinematic_constraints_.data(),
                   enabled_joint_kinematic_constraints_.size() - 1)),
               absl::OkStatus())
          << "Cartesian6dToJointVelocityMapper::UpdateConstraints: The "
             "joint position limit and joint velocity limit constraints "
             "are in conflict after removing the joint acceleration "
             "constraint ["
          << kinematic_constraints_status.ToString()
          << "]. This should never happen; please contact "
             "the developers for more information.";
    }
  }

  // Update collision avoidance.
  if (has_collision_avoidance_constraint) {
    collision_avoidance_constraint_->UpdateCoefficientsAndBounds(*data_);
  }
  if (log_collision_warnings_) {
    auto contact_debug_str =
        collision_avoidance_constraint_->GetContactDebugString(*data_);
    LOG_IF(WARNING, !contact_debug_str.empty()) << contact_debug_str;
  }
  return absl::OkStatus();
}

// Assumes solution_ has the computed values to be checked for validity.
absl::Status Cartesian6dToJointVelocityMapper::CheckSolutionValidity() {
  // If collision avoidance is not active, no need to check.
  if (collision_avoidance_constraint_ == nullptr) {
    return absl::OkStatus();
  }

  // If collision avoidance is active, do a sanity check to ensure that the
  // integrated velocities do not increase penetration. This is because the
  // computed velocities may result in penetration after integration, even if
  // they are in the feasible space.
  //
  // At the moment, we assume that all joints that are not being controlled are
  // fixed, but the collision avoidance constraint may be able to support other
  // moving joints in the future.
  //
  // We have identified 2 main reasons why this can happen:
  //   * Errors due to the local Jacobian linearization about the current
  //     configuration;
  //   * MuJoCo's collision detection is sometimes flaky between certain geom
  //     types, and the computed normal distance/direction may be wrong.
  ComputeGeomPairDist(lib_, model_, *data_, minimum_distance_from_collisions_,
                      &geom_pair_to_dist_curr_);
  std::fill_n(data_->qvel, model_.nv, 0.0);  // Assume everything else is fixed.
  IntegrateQpos(lib_, model_, integration_timestep_, joint_dof_ids_, solution_,
                data_.get());
  lib_.mj_kinematics(&model_, data_.get());
  ComputeGeomPairDist(lib_, model_, *data_, minimum_distance_from_collisions_,
                      &geom_pair_to_dist_after_);
  std::pair<int, int> penetrating_pair;
  if (IsPenetrationIncreased(minimum_distance_from_collisions_,
                             geom_pair_to_dist_curr_, geom_pair_to_dist_after_,
                             &penetrating_pair)) {
    // The current distance may or may not have a value, as it could be the case
    // that no collision was detected before and then suddenly it was
    // penetrating. If that's the case, we instead print
    // ">[`min_distance_from_collision`]".
    std::optional<double> dist_before =
        geom_pair_to_dist_curr_.at(penetrating_pair);
    std::string dist_before_msg =
        dist_before.has_value()
            ? std::to_string(dist_before.value())
            : absl::StrCat(">", minimum_distance_from_collisions_);

    return absl::InternalError(absl::Substitute(
        "Cartesian6dToJointVelocityMapper::ComputeJointVelocities: Computed "
        "joint velocities resulted in increased penetration when integrated. "
        "This can happen if the `collision_detection_distance` is too small, "
        "`collision_avoidance_normal_velocity_scale` is too large, or if "
        "MuJoCo's collision detection mechanism outputs the wrong normal or "
        "distance between geoms. Increasing `collision_detection_distance` and "
        "decreasing `collision_avoidance_normal_velocity_scale` may help. "
        "Specifically, normal distance between geom with name [$0] and geom "
        "with name [$1] decreased from a current value of [$2] to a value of "
        "[$3] after integration, with a user-defined minimum distance from "
        "collisions of [$4].",
        lib_.mj_id2name(&model_, mjOBJ_GEOM, penetrating_pair.first),
        lib_.mj_id2name(&model_, mjOBJ_GEOM, penetrating_pair.second),
        dist_before_msg, *geom_pair_to_dist_after_.at(penetrating_pair),
        minimum_distance_from_collisions_));
  }
  return absl::OkStatus();
}

}  // namespace dm_robotics
