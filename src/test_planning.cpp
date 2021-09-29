#include <ros/ros.h>
#include <graph_core/solvers/path_local_solver.h>
#include <graph_core/solvers/path_solver.h>
#include <graph_core/solvers/birrt.h>
#include <graph_core/solvers/rrt_star.h>
#include <graph_core/parallel_moveit_collision_checker.h>
#include <graph_core/metrics.h>
#include <graph_core/local_informed_sampler.h>
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

bool computePath(ros::NodeHandle& nh, const Eigen::VectorXd& start_conf, const Eigen::VectorXd& goal_conf, pathplan::PathPtr& solution, pathplan::TreeSolverPtr solver)
{
  pathplan::NodePtr start_node = std::make_shared<pathplan::Node>(start_conf);
  pathplan::NodePtr goal_node = std::make_shared<pathplan::Node>(goal_conf);

  solver->config(nh);
  solver->addStart(start_node);
  solver->addGoal(goal_node);

  bool success = true;
  if (!solver->solve(solution, 1000000))
  {
    ROS_INFO("No solutions found");
    return false;
  }

  if(success)
  {
    pathplan::PathLocalOptimizer path_solver(solver->getChecker(), solver->getMetrics());
    path_solver.config(nh);

    solution->setTree(solver->getStartTree());
    path_solver.setPath(solution);
    path_solver.solve(solution);

    solver->getSampler()->setCost(solution->cost());
    solver->getStartTree()->addBranch(solution->getConnections());

    pathplan::LocalInformedSamplerPtr local_sampler = std::make_shared<pathplan::LocalInformedSampler>(start_node->getConfiguration(), goal_node->getConfiguration(), solver->getSampler()->getLB(), solver->getSampler()->getUB());

    for (unsigned int isol = 0; isol < solution->getConnections().size() - 1; isol++)
    {
      pathplan::ConnectionPtr conn = solution->getConnections().at(isol);
      local_sampler->addBall(conn->getChild()->getConfiguration(), solution->cost() * 0.1);
    }
    local_sampler->setCost(solution->cost());

    pathplan::RRTStar opt_solver(solver->getMetrics(), solver->getChecker(), local_sampler);
    opt_solver.addStartTree(solver->getStartTree());
    opt_solver.addGoal(goal_node);
    opt_solver.config(nh);

    std::vector<pathplan::NodePtr> white_list;
    white_list.push_back(goal_node);

    ros::Duration max_time(3);
    ros::Time t0 = ros::Time::now();

    int stall_gen = 0;
    int max_stall_gen = 2000;

    std::mt19937 gen;
    std::uniform_int_distribution<> id = std::uniform_int_distribution<>(0, max_stall_gen);

    for (unsigned int idx = 0; idx < 100000; idx++)
    {
      if (ros::Time::now() - t0 > max_time)
        break;

      if (opt_solver.update(solution))
      {
        stall_gen = 0;
        path_solver.setPath(solution);
        solution->setTree(opt_solver.getStartTree());

        local_sampler->setCost(solution->cost());
        solver->getSampler()->setCost(solution->cost());
        opt_solver.getStartTree()->purgeNodes( solver->getSampler(), white_list, true);

        local_sampler->clearBalls();
        for (unsigned int isol = 0; isol < solution->getConnections().size() - 1; isol++)
        {
          pathplan::ConnectionPtr conn = solution->getConnections().at(isol);

          local_sampler->addBall(conn->getChild()->getConfiguration(), solution->cost() * 0.1);
        }
      }
      else
      {
        opt_solver.getStartTree()->purgeNodes( solver->getSampler(), white_list, false);
        stall_gen++;
      }

      if (idx % 10 == 0)

        if (id(gen) < stall_gen)
        {
          opt_solver.setSampler( solver->getSampler());
        }
        else
        {
          opt_solver.setSampler(local_sampler);
        }

      if (stall_gen >= max_stall_gen)
        break;
    }

    path_solver.setPath(solution);
    path_solver.solve(solution);
  }

  return true;
}

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

  int fail = 0;

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

    ROS_INFO_STREAM("START CONf COLLISION: "<< start_in_collision);
    ROS_INFO_STREAM("GOAL CONF COLLISION: "<< goal_in_collision);

    if(goal_in_collision || start_in_collision)
     {
      return 0;
    }

    for(unsigned int i=0;i<n_repetitions;i++)
    {
      ROS_INFO_STREAM("Query: "<<query<< " repetition: "<<i);
      pathplan::SamplerPtr sampler = std::make_shared<pathplan::InformedSampler>(start_conf, goal_conf, lb, ub);
      pathplan::BiRRTPtr solver = std::make_shared<pathplan::BiRRT>(metrics, checker, sampler);

      pathplan::PathPtr current_path;

      if(!computePath(nh,start_conf,goal_conf,current_path,solver))
      {
        ROS_WARN("Solution not found skip!");
        fail+=1;
        continue;
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

  ROS_INFO_STREAM("FAIL: "<<fail);
  return 0;
}

