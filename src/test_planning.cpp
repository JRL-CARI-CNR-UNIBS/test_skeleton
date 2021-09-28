#include <ros/ros.h>
#include <graph_core/solvers/birrt.h>
#include <graph_core/solvers/rrt_star.h>
#include <graph_core/parallel_moveit_collision_checker.h>
#include <graph_core/metrics.h>
#include <graph_core/parallel_moveit_collision_checker.h>
#include <graph_core/graph/graph_display.h>
#include <moveit/robot_state/robot_state.h>
#include <object_loader_msgs/AddObjects.h>
#include <object_loader_msgs/RemoveObjects.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <rviz_visual_tools/rviz_visual_tools.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_planning");
  ros::AsyncSpinner spinner(4);
  spinner.start();

  ros::NodeHandle nh;

  //  ROS_INFO("WAITING FOR ROSBAG..");
  //  ros::Duration(20).sleep();
  std::vector<double> start_configuration;

  std::string kind_test;
  if (!nh.getParam("kind_test",kind_test))
  {
    ROS_ERROR("kind_test not set, exit");
    return 0;
  }

  int n_repetitions;
  if (!nh.getParam("n_repetitions",n_repetitions))
  {
    ROS_ERROR("n_repetitions not set,exit");
    return 0;
  }

  int n_query;
  if (!nh.getParam("n_query",n_query))
  {
    ROS_ERROR("n_query not set, exit");
    return 0;
  }

  int start_query;
  if (!nh.getParam("start_query",start_query))
  {
    ROS_ERROR("start_query not set, exit");
    return 0;
  }

  std::string topic_name = "/test_planning_";
  topic_name.append(kind_test);
  ros::Publisher test_planning_pub = nh.advertise<std_msgs::Float64MultiArray>(topic_name,1);

  // /////////////////////////////////UPLOADING THE ROBOT ARM/////////////////////////////////////////////////////////////
  std::string group_name = "manipulator";
  std::string base_link = "base_link";
  std::string last_link = "open_tip";
  moveit::planning_interface::MoveGroupInterface move_group(group_name);
  robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
  robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();
  planning_scene::PlanningScenePtr planning_scene = std::make_shared<planning_scene::PlanningScene>(kinematic_model);

  const robot_state::JointModelGroup* joint_model_group = move_group.getCurrentState()->getJointModelGroup(group_name);
  std::vector<std::string> joint_names = joint_model_group->getActiveJointModelNames();

  unsigned int dof = joint_names.size();
  Eigen::VectorXd lb(dof);
  Eigen::VectorXd ub(dof);

  for (unsigned int idx = 0; idx < dof; idx++)
  {
    const robot_model::VariableBounds& bounds = kinematic_model->getVariableBounds(joint_names.at(idx));
    if (bounds.position_bounded_)
    {
      lb(idx) = bounds.min_position_;
      ub(idx) = bounds.max_position_;
    }
  }

  // ////////////////////////////////////////UPDATING THE PLANNING SCENE////////////////////////////////////////////////////////
  ros::ServiceClient ps_client=nh.serviceClient<moveit_msgs::GetPlanningScene>("/get_planning_scene");
  moveit_msgs::GetPlanningScene ps_srv;

  if (!ps_client.call(ps_srv))
  {
    ROS_ERROR("call to srv not ok");
    return 0;
  }

  if (!planning_scene->setPlanningSceneMsg(ps_srv.response.scene))
  {
    ROS_ERROR("unable to update planning scene");
    return 0;
  }
  // //////////////////////////////////////////PATH PLAN & VISUALIZATION////////////////////////////////////////////////////////
  pathplan::MetricsPtr metrics = std::make_shared<pathplan::Metrics>();
  pathplan::CollisionCheckerPtr checker = std::make_shared<pathplan::MoveitCollisionChecker>(planning_scene, group_name);

  pathplan::DisplayPtr disp = std::make_shared<pathplan::Display>(planning_scene,group_name,last_link);
  disp->clearMarkers();
  ros::Duration(1).sleep();

  for(unsigned int j=start_query;j<n_query;j++)
  {
    std::string query = "query_";
    query = query+std::to_string(j);

    if (!nh.getParam(query+"/start_configuration",start_configuration))
    {
      ROS_ERROR("start_configuration not set, exit");
      return 0;
    }
    std::vector<double> stop_configuration;
    if (!nh.getParam(query+"/stop_configuration",stop_configuration))
    {
      ROS_ERROR("stop_configuration not set, exit");
      return 0;
    }

    std::vector<pathplan::PathPtr> path_vector;
    Eigen::VectorXd start_conf = Eigen::Map<Eigen::VectorXd>(start_configuration.data(), start_configuration.size());
    Eigen::VectorXd goal_conf = Eigen::Map<Eigen::VectorXd>(stop_configuration.data(), stop_configuration.size());

    start_conf = start_conf*3.14159/180.0;
    goal_conf =  goal_conf*3.14159/180.0;

    bool start_in_collision = !checker->check(start_conf);
    bool goal_in_collision = !checker->check(goal_conf);

    ROS_INFO_STREAM("START CONF: "<<start_conf.transpose()<< " COLLISION: "<< start_in_collision);
    ROS_INFO_STREAM("GOAL CONF: "<<goal_conf.transpose()<< " COLLISION: "<< goal_in_collision);

    for(unsigned int i=0;i<n_repetitions;i++)
    {
      ROS_INFO_STREAM("Query: "<<query<< " repetition: "<<i);
      pathplan::SamplerPtr sampler = std::make_shared<pathplan::InformedSampler>(start_conf, goal_conf, lb, ub);
      pathplan::BiRRTPtr solver = std::make_shared<pathplan::BiRRT>(metrics, checker, sampler);

      pathplan::PathPtr current_path;

      solver->config(nh);
      solver->addStart(std::make_shared<pathplan::Node>(start_conf));
      solver->addGoal(std::make_shared<pathplan::Node>(goal_conf));

      if (!solver->solve(current_path, 100000))
      {
        ROS_INFO("No solutions found");
      }

      double cost2sent = current_path->cost();

      std::vector<double> marker_color = {0.5,0.5,0.0,1.0};
      disp->displayPathAndWaypoints(current_path,3473,12189,"pathplan",marker_color);

      std_msgs::Float64MultiArray msg;
      msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
      msg.layout.dim[0].size = 15;
      msg.layout.dim[0].stride = 1;
      msg.layout.dim[0].label = query+"/"+std::to_string(i);
      msg.data.push_back(j);
      msg.data.push_back(i);
      msg.data.insert(msg.data.end(),start_configuration.begin(),start_configuration.end());
      msg.data.insert(msg.data.end(),stop_configuration.begin(),stop_configuration.end());
      msg.data.push_back(cost2sent);

      test_planning_pub.publish(msg);

      ROS_INFO_STREAM("Solution cost: "<<cost2sent);
      ros::Duration(5.0).sleep();
    }
  }
  return 0;
}

