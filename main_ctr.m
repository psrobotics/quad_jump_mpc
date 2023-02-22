% Quadruped jumping ctr demo, with linear srb model

%% pre clear
clc;
clear;
close all;
warning off;

addpath('D:\matlab_lib\casadi-windows-matlabR2016a-v3.5.5');
import casadi.*;

%% pre-defined values for controller
phase_num = 4;
N = 80; % mpc step window
T = 1.3; % sim time
dt_val = (T/N) .* ones(N,1); % dt vector

max_jump_z = 0.55; % max jumping height
min_dump_z = 0.15; % min standing height
max_lift_vel_z = 6.5; % max jumping velocity
init_z = 0.3; % init standing height

x_init_tar_val = [0; 0; 0; 0; 0; init_z]; % init state
dx_init_tar_val = [0; 0; 0; 0; 0; 0]; % init d state
x_final_tar_val = [0; -pi*0.3*0; pi*0.5; 1.5; 0; 0.2]; % final target state
dx_final_tar_val = [0; 0; 0; 0; 0; 0];

contact_state_ratio = N.*[0.35 0.15 0.475 0.025]; % pull, jump, flight, impact
contact_state_val = [ones(4, contact_state_ratio(1)),...
                     ones(4, contact_state_ratio(2)),...
                     0 * ones(4, contact_state_ratio(3)),...
                     0 * ones(4, contact_state_ratio(4))]; % no foot contact during last 2 phases
                 
% cost gains
weight.QX = [10 10 10, 10 10 10, 10 10 10, 10 10 10 ]';
weight.QN = [10 10 10, 50 50 50, 10 10 10, 10 10 10 ]';
weight.Qc = 1*[0.01 0.01 0.01]';
weight.Qf = [0.0001 0.0001 0.001]';
                 
%% SRB dynamic model
world.fk = 0.5; %friction coefficient
world.g = 9.81;

body.m = 5*2;

i_vec = [0.059150 0.101150 0.046240]*2;
body.i_mat = [i_vec(1) 0 0;... % roll
           0 i_vec(2) 0;... % pitch
           0 0 i_vec(3)]; % yaw
body.length = 0.34;
body.width = 0.26;

% foot motion range
body.foot_x_range = 0.15;
body.foot_y_range = 0.15;
body.foot_z_range = 0.3;

body.max_zforce = 1000;

hip_vec = [body.length/2; body.width/2; 0];
hip_dir_mat = [1 1 -1 -1; 1 -1 1 -1; 0 0 0 0];
body.hip_pos = hip_dir_mat .* repmat(hip_vec,1,4);
body.foot_pos = repmat([0; 0; -0.6*init_z],1,4); % init foot pos

body.phip_swing_ref = body.hip_pos + body.foot_pos;
body.phip_swing_ref_vec = reshape(body.phip_swing_ref,[],1); % ref foot pos at swing phase

% build the dynamic equation
state_dim = 12; % number of dim of state, rpy xyz
f_dim = 12; % number of dim of leg force, 3*4
fp_dim = 12; % number of dim of leg pos, 3*4

% casadi variables
x_k = SX.sym('x_k', state_dim, 1); % state
f_k = SX.sym('f_k', f_dim, 1); % foot force
fp_k = SX.sym('fp_k', fp_dim, 1); % foot position

% z-psi-yaw
% y-theta-pitch
% x-phi-roll

rot_mat_zyx = rot_zyx(x_k(1:3));

s_yaw = sin(x_k(3));
c_yaw = cos(x_k(3));
t_yaw = tan(x_k(3));

s_pitch = sin(x_k(2));
c_pitch = cos(x_k(2));
t_pitch = tan(x_k(2));

inv_rot_linear = [c_yaw s_yaw 0; 
                  -1*s_yaw c_yaw 0; 
                  0 0 1];
              
inv_rot_nonlinear = [c_yaw/c_pitch s_yaw/c_pitch 0;
                     -1*s_yaw c_yaw 0;
                     c_yaw*t_pitch s_yaw*t_pitch 1];
                
% convert the intertia tensor from local cod to world cod
i_mat_w = rot_mat_zyx*body.i_mat*rot_mat_zyx'; %i_mat in world
i_mat_w_inv = eye(3)/i_mat_w;

% A, B, G mat
A = [zeros(3) zeros(3) inv_rot_linear zeros(3);...
     zeros(3) zeros(3) zeros(3) eye(3);...
     zeros(3) zeros(3) zeros(3) zeros(3);...
     zeros(3) zeros(3) zeros(3) zeros(3)];
% A = [z3 z3 inv_rot z3;
%      z3 z3 z3      I3
%      z3 z3 z3      z3
%      z3 z3 z3      z3];

B = [zeros(3) zeros(3) zeros(3) zeros(3);...
     zeros(3) zeros(3) zeros(3) zeros(3);...
     i_mat_w_inv*skew_mat(fp_k(1:3)), i_mat_w_inv*skew_mat(fp_k(4:6)), i_mat_w_inv*skew_mat(fp_k(7:9)), i_mat_w_inv*skew_mat(fp_k(10:12));...
     eye(3)/body.m, eye(3)/body.m, eye(3)/body.m, eye(3)/body.m];
% B = [z3 z3 z3 z3
%      z3 z3 z3 z3
%      (I_m)^-1*[f_pos]x
%      I3/m I3/m I3/m I3/m];

G = zeros(12,1);
G(12) = -1*world.g;

d_x = A*x_k + B*f_k + G;

% map the dynamic function
dyn_f = Function('dyn_f',{x_k;f_k;fp_k},{d_x},{'state','leg_force','foot_pos'},{'d_state'});

%x_init = [0;0.0;0; 0.0;0.0;0.5 ;0;0;0; 0;0;0];
%dyn_f(x_init,zeros(12,1),zeros(12,1))

%% Variables for costs and constraints
% casadi variables array for the optimal window
x_arr = SX.sym('x_arr', state_dim, N+1); % state
f_arr = SX.sym('f_arr', f_dim, N); % foot force
fp_arr = SX.sym('fp_arr', fp_dim, N); % foot position

% casadi variables for the reference variables
x_ref_arr = SX.sym('x_ref_arr', state_dim, N+1); % state
f_ref_arr = SX.sym('f_ref_arr', f_dim, N); % foot force
fp_ref_arr = SX.sym('fp_ref_arr', fp_dim, N); % foot position

% contact mat arr, wheter the leg touches ground
contact_mat_arr = SX.sym('contact_mat_arr', 4, N);

friction_cone = [1/world.fk, 0 -1;...
                 -1/world.fk, 0 -1;...
                 0, 1/world.fk, -1;...
                 0, -1/world.fk, -1];

foot_convex_hull = [1 0 0 -body.foot_x_range;
                    -1 0 0 -body.foot_x_range;
                    0 1 0 -body.foot_y_range;
                    0 -1 0 -body.foot_y_range;
                    0 0 1 -min_dump_z;
                    0 0 -1 -body.foot_z_range];
                
% eq constraints
eq_con_init_state = x_ref_arr(:,1)-x_arr(:,1); %init condition constraint, 12,1

% init other constraints with casadi variables
eq_con_dynamic = SX.zeros(state_dim*(N+1),1); % dynamic constraint, 12*(N+1),1
eq_con_foot_contact_ground = SX.zeros(4*N,1); % constraint on leg's motion range, 4*N,1
eq_con_foot_non_slip = SX.zeros(fp_dim*N,1); % constraint on foot's non slip stane phase, 12*N,1

eq_con_dim = 12 + state_dim*(N+1) + 4*N + fp_dim*N; % dim for eq constraints

% ieq constraints
ieq_con_foot_range = SX.zeros(4*6*N,1); % ieq constraint on foot's motion range 4leg*3axis*2dir
ieq_con_foot_friction = SX.zeros(4*4*N,1); % ieq constraint on foot friction 4*2axis*2dir
ieq_con_zforce_dir = SX.zeros(1*N,1); % z axis force always pointing up
ieq_con_zforce_range = SX.zeros(4*N,1); % z axis force range, swing phase ->0, stance phase -> < max z force

ieq_con_dim = 4*6*N + 4*4*N + N + 4*N; % dim for ieq constraints

cost_fcn = 0;

% define cost function and constraints
for k = 1:N
    x_t = x_arr(:,k);
    x_t_next = x_arr(:,k+1);
    
    f_t = f_arr(:,k);
    fp_t = fp_arr(:,k);
    
    fp_g_t = repmat(x_arr(4:6,k),4,1)+fp_arr(:,k); %foot pos in global coord
    
    x_ref_t = x_ref_arr(:,k);
    f_ref_t = f_ref_arr(:,k);
    fp_ref_t = fp_ref_arr(:,k);
    
    contact_mat_t = contact_mat_arr(:,k);
    dt_t = dt_val(k);
    
    x_err = x_t - x_ref_t; % state error
    f_err = f_t - f_ref_t; % leg force error
    fp_err = repmat(x_t(4:6),4,1) + body.phip_swing_ref_vec - fp_g_t; % foot pos error
    
    % state cost
    cost_fcn = cost_fcn + (x_err'*diag(weight.QX)*x_err...
                           + f_err'*diag(repmat(weight.Qf,4,1))*f_err...
                           + fp_err'*diag(repmat(weight.Qc,4,1))*fp_err) * dt_t;
  
    % constraints
    % dynamic equation constraint
    eq_con_dynamic(state_dim*(k-1)+1:state_dim*k) = x_t_next - (x_t + dyn_f(x_t,f_t,fp_t)*dt_t);
    % zforce direction
    ieq_con_zforce_dir(k)=-1*dot(f_t,repmat([0;0;1],4,1));
    % zforce range
    ieq_con_zforce_range(4*(k-1)+1:4*k) = f_t([3,6,9,12]) - contact_mat_t.*repmat(body.max_zforce,4,1);
    
    for leg_k = 1:4
        xyz_k = 3*(leg_k-1)+1:3*leg_k;
        
        % constrant when leg on ground
        eq_con_foot_contact_ground((k-1)*4+leg_k) = contact_mat_t(leg_k)*fp_g_t(3*(leg_k-1)+3);
        
        % foot motion range
        rot_zyx_t = rot_zyx(x_t(1:3));
        phip_global_t = rot_zyx_t*body.phip_swing_ref + x_t(4:6);
        leg_vec_t = (fp_g_t(xyz_k) - phip_global_t(:,leg_k)); % leg vector, from hip to foot
        ieq_con_foot_range(24*(k-1)+6*(leg_k-1)+1: 24*(k-1)+6*leg_k) = foot_convex_hull*[leg_vec_t;1]; % leg's motion range limit
        
        % friction cone
        ieq_con_foot_friction(16*(k-1)+4*(leg_k-1)+1: 16*(k-1)+4*leg_k) = friction_cone*f_t(xyz_k);
        
        % non-slip 
        if (k < N)
            fp_g_t_next = repmat(x_arr(4:6,k+1),4,1)+fp_arr(:,k+1);
            eq_con_foot_non_slip(12*(k-1)+3*(leg_k-1)+1: 12*(k-1)+3*leg_k) = contact_mat_t(leg_k)*(fp_g_t_next(xyz_k)-fp_g_t(xyz_k));
        end
    end
    
end

% add final state cost
x_err_final = x_arr(:,N+1)-x_ref_arr(:,N+1);
cost_fcn = cost_fcn + x_err_final'*diag(weight.QN)*x_err_final;

% combine all constraints
con_arr = [eq_con_init_state; eq_con_dynamic; eq_con_foot_contact_ground; eq_con_foot_non_slip;...
           ieq_con_foot_range; ieq_con_foot_friction; ieq_con_zforce_dir; ieq_con_zforce_range];
       
%% init the opt problem and solver
% reform into vector, state, leg force, foot pos
opt_variables = [reshape(x_arr,state_dim*(N+1),1); reshape(f_arr,f_dim*N,1); reshape(fp_arr,fp_dim*N,1)];
% reference state, leg force, foot pos, contact event matrix
opt_ref_param = [reshape(x_ref_arr,state_dim*(N+1),1); reshape(f_ref_arr,f_dim*N,1); reshape(fp_ref_arr,fp_dim*N,1); reshape(contact_mat_arr,4*N,1)];
nlp_prob = struct('f',cost_fcn, 'x',opt_variables, 'p',opt_ref_param, 'g',con_arr);

%% optimal settings
opt_setting.expand =true;
opt_setting.ipopt.max_iter=1500;
opt_setting.ipopt.print_level=0;
opt_setting.ipopt.acceptable_tol=1e-4;
opt_setting.ipopt.acceptable_obj_change_tol=1e-6;
opt_setting.ipopt.tol=1e-4;
opt_setting.ipopt.nlp_scaling_method='gradient-based';
opt_setting.ipopt.constr_viol_tol=1e-3;
opt_setting.ipopt.fixed_variable_treatment='relax_bounds';

%% solver
solver = nlpsol('solver','ipopt',nlp_prob,opt_setting);

%% lower & upper boundary of eq & ieq constrains
args.lbg(1:eq_con_dim) = 0;
args.ubg(1:eq_con_dim) = 0;

args.lbg(eq_con_dim+1: eq_con_dim+ieq_con_dim) = -inf;
args.ubg(eq_con_dim+1: eq_con_dim+ieq_con_dim) = 0;

% state lower & upper boundary
% state
ub_state = [3*pi*ones(3,1);10*ones(2,1);max_jump_z;...
            5*pi*ones(3,1);50*ones(2,1);max_lift_vel_z];
lb_state = [-3*pi*ones(3,1);-10*ones(2,1);min_dump_z;...
            -5*pi*ones(3,1);-50*ones(3,1)];
ub_state_arr = repmat(ub_state,N+1,1);
lb_state_arr = repmat(lb_state,N+1,1);

% leg force
ub_f_leg = [body.m*world.g*world.fk*50; body.m*world.g*world.fk*50; body.max_zforce]; %xyz maximum leg force
lb_f_leg = [-1*body.m*world.g*world.fk*50; -1*body.m*world.g*world.fk*50; 0]; %minimum leg force
ub_f = repmat(ub_f_leg,4,1); % for 4 legs
lb_f = repmat(lb_f_leg,4,1);
ub_f_arr = repmat(ub_f,N,1);
lb_f_arr = repmat(lb_f,N,1);

% foot pos
ub_fp = repmat([0.4;0.4;inf],4,1);
lb_fp = repmat([-0.4;-0.4;-inf],4,1);
ub_fp_arr = repmat(ub_fp,N,1);
lb_fp_arr = repmat(lb_fp,N,1);

% combine lower & upper bounds together
args.ubx = [ub_state_arr; ub_f_arr; ub_fp_arr];
args.lbx = [lb_state_arr; lb_f_arr; lb_fp_arr];

%% Generate reference trajectory
fp_ref_i = diag([1 1 1, 1 -1 1, -1 1 1, -1 -1 1])*repmat([0.4 0.2 -1*init_z],1,4)'; % foot pos ref

x_ref_val = zeros(12,N+1);
f_ref_val = zeros(12,N);
fp_ref_val = zeros(12,N);

for i = 1:6
    x_ref_val(i,:) = linspace(x_init_tar_val(i),x_final_tar_val(i),N+1); %rpy xyz
    x_ref_val(i+6,:) = linspace(dx_init_tar_val(i),dx_final_tar_val(i),N+1); %velocity
end

% spine on the z axis
s_a = [x_ref_val(4,1),x_ref_val(4,N/2),x_ref_val(4,N)]; % x axis
s_b = [x_init_tar_val(6),x_final_tar_val(6),x_init_tar_val(6)+0]; % z axis
x_ref_val(6,:) = interp1(s_a,s_b,x_ref_val(4,:),'spline');


for leg_i = 1:4
    for xyz_j = 1:3
        %u_ref_val(3*(leg_i-1)+xyz_j,:) = x_ref_val(xyz_j:end-1) + fp_ref_i(3*(leg_i-1)+xyz_j);
        fp_ref_val(3*(leg_i-1)+xyz_j,:) = fp_ref_i(3*(leg_i-1)+xyz_j);
    end
end

% combine all ref traj
args.p = [reshape(x_ref_val,state_dim*(N+1),1);...
          reshape(f_ref_val,f_dim*N,1);...
          reshape(fp_ref_val,fp_dim*N,1);...
          reshape(contact_state_val,4*N,1)];
      
args.x0 = [reshape(x_ref_val,state_dim*(N+1),1);...
          reshape(f_ref_val,f_dim*N,1);...
          reshape(fp_ref_val,fp_dim*N,1)]; % initial states


%% Slove the NLP prob
sol = solver('x0',args.x0, 'lbx',args.lbx, 'ubx',args.ubx, 'lbg',args.lbg, 'ubg',args.ubg, 'p',args.p);

%% Vis
x_li=sol.x(1:state_dim*(N+1));
x_li=reshape(full(x_li),state_dim,(N+1));% COM under world coord

f_li=sol.x(state_dim*(N+1)+1:state_dim*(N+1)+f_dim*N);
f_li=reshape(full(f_li),f_dim,N);% foot force

r_li=sol.x(state_dim*(N+1)+f_dim*N+1:state_dim*(N+1)+f_dim*N+fp_dim*N);
r_li=reshape(full(r_li),f_dim,N);% Foot pos in rbt coord
p_li=r_li+repmat(x_li(4:6,1:end-1),4,1);% Foot pos in world coord

rbt_visualization(x_li,f_li,p_li,[],T,N);













