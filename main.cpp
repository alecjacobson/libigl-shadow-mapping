#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/create_shader_program.h>
#include <igl/opengl/destroy_shader_program.h>
#include <igl/opengl/report_gl_error.h>
#include <igl/look_at.h>
#include <igl/ortho.h>
#include <igl/null.h>
#include <igl/unique.h>
#include <igl/PI.h>
#include <igl/get_seconds.h>
#include <igl/png/writePNG.h>

template <typename T>
void read_pixels(
  const GLuint width,
  const GLuint height,
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> & R,
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> & G,
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> & B,
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> & A,
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> & D)
{
  R.resize(width,height);
  G.resize(width,height);
  B.resize(width,height);
  A.resize(width,height);
  D.resize(width,height);
  typedef typename std::conditional< std::is_floating_point<T>::value,GLfloat,GLubyte>::type GLType;
  GLenum type = std::is_floating_point<T>::value ?  GL_FLOAT : GL_UNSIGNED_BYTE;
  GLType* pixels = (GLType*)calloc(width*height*4,sizeof(GLType));
  GLType * depth = (GLType*)calloc(width*height*1,sizeof(GLType));
  glReadPixels(0, 0,width, height,GL_RGBA,            type, pixels);
  glReadPixels(0, 0,width, height,GL_DEPTH_COMPONENT, type, depth);
  int count = 0;
  for (unsigned j=0; j<height; ++j)
  {
    for (unsigned i=0; i<width; ++i)
    {
      R(i,j) = pixels[count*4+0];
      G(i,j) = pixels[count*4+1];
      B(i,j) = pixels[count*4+2];
      A(i,j) = pixels[count*4+3];
      D(i,j) = depth[count*1+0];
      ++count;
    }
  }
  // Clean up
  free(pixels);
  free(depth);
}

int main(int argc, char *argv[])
{
  // Inline mesh of a cube
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::read_triangle_mesh(argv[1],V,F);
  // Plot the mesh
  igl::opengl::glfw::Viewer vr;
  vr.core().light_position << 1,0,0;
  //vr.core().animation_max_fps = 20;
  vr.core().is_animating = true;
  vr.data().set_mesh(V, F);
  vr.data().set_face_based(true);
  vr.data().show_lines = false;
  vr.launch_init(true,false);
  vr.data().meshgl.init();

  // Initialize a shadow map data structures
  GLuint shadow_depth_tex;
  GLuint shadow_depth_fbo;
  GLuint shadow_color_rbo;
  GLuint shadow_width =  1024;
  GLuint shadow_height = 1024;
  printf("\n");
  igl::opengl::report_gl_error("before shadow map initialization");
  {
    printf("shadow_depth_tex: %d\n",shadow_depth_tex);
    // Create a texture for writing the shadow map depth values into
    {
      glDeleteTextures(1,&shadow_depth_tex);
      glGenTextures(1, &shadow_depth_tex);
      glBindTexture(GL_TEXTURE_2D, shadow_depth_tex);
      // Should this be using double/float precision?
      glTexImage2D(
        GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
        shadow_width,
        shadow_height,
        0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glBindTexture(GL_TEXTURE_2D, 0);
    }
    printf("shadow_depth_tex: %d\n",shadow_depth_tex);
    // Generate a framebuffer with depth attached to this texture and color
    // attached to a render buffer object
    glGenFramebuffers(1, &shadow_depth_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_depth_fbo);
    // Attach depth texture
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
        shadow_depth_tex,0);
    // Generate a render buffer to write colors into. Low precision we don't
    // care about them. Is there a way to not write/compute them at? Probably
    // just need to change fragment shader.
    glGenRenderbuffers(1,&shadow_color_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER,shadow_color_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, shadow_width, shadow_height);
    // Attach color buffer
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_RENDERBUFFER, shadow_color_rbo);
    //Does the GPU support current FBO configuration?
    GLenum status;
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    switch(status)
    {
      case GL_FRAMEBUFFER_COMPLETE:
        break;
      default:
        printf("We failed to set up a good FBO: %d\n",status);
        assert(false);
    }
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }
  igl::opengl::report_gl_error("after shadow map initialization");

  igl::opengl::destroy_shader_program(vr.data().meshgl.shader_mesh);


  std::string mesh_vertex_shader_string =
R"(#version 150
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;
  in vec3 position;
  in vec3 normal;
  out vec3 position_eye;
  out vec3 normal_eye;
  in vec4 Ka;
  in vec4 Kd;
  in vec4 Ks;
  in vec2 texcoord;
  out vec2 texcoordi;
  out vec4 Kai;
  out vec4 Kdi;
  out vec4 Ksi;
  uniform mat4 shadow_view;
  uniform mat4 shadow_proj;
  uniform bool shadow_pass;
  out vec4 position_shadow;

  void main()
  {
    position_eye = vec3 (view * vec4 (position, 1.0));
    if(!shadow_pass)
    {
      position_shadow = shadow_proj * shadow_view * vec4(position, 1.0);
      normal_eye = vec3 (normal_matrix * vec4 (normal, 0.0));
      normal_eye = normalize(normal_eye);
      Kai = Ka;
      Kdi = Kd;
      Ksi = Ks;
      texcoordi = texcoord;
    }
    gl_Position = proj * vec4 (position_eye, 1.0); //proj * view * vec4(position, 1.0);"
  }
)";

  std::string mesh_fragment_shader_string =
R"(#version 150
  uniform mat4 view;
  uniform mat4 proj;
  uniform vec4 fixed_color;
  in vec3 position_eye;
  in vec3 normal_eye;
  uniform vec3 light_position_eye;
  vec3 Ls = vec3 (1, 1, 1);
  vec3 Ld = vec3 (1, 1, 1);
  vec3 La = vec3 (1, 1, 1);
  in vec4 Ksi;
  in vec4 Kdi;
  in vec4 Kai;
  in vec2 texcoordi;
  uniform sampler2D tex;
  uniform float specular_exponent;
  uniform float lighting_factor;
  uniform float texture_factor;
  uniform float matcap_factor;
  uniform float double_sided;

  uniform bool shadow_pass;
  uniform sampler2D shadow_tex;
  in vec4 position_shadow;

  out vec4 outColor;
  void main()
  {
    if(shadow_pass)
    {
      outColor = vec4(0.56,0.85,0.77,1.);
      return;
    }
    if(matcap_factor == 1.0f)
    {
      vec2 uv = normalize(normal_eye).xy * 0.5 + 0.5;
      outColor = texture(tex, uv);
    }else
    {
      //vec3 vector_to_light_eye = light_position_eye - position_eye;
      // Interpret light_position_eye as direction
      vec3 vector_to_light_eye = light_position_eye;
      vec3 direction_to_light_eye = normalize (vector_to_light_eye);

      vec3 Ia = La * vec3(Kai);    // ambient intensity

      vec3 shadow_proj = (position_shadow.xyz / position_shadow.w) * 0.5 + 0.5; 
      float currentDepth = shadow_proj.z;
      
      //float bias = 0.005;
      float ddd = max(dot(normalize(normal_eye), direction_to_light_eye),0);
      float bias = max(0.02 * (1.0 - ddd), 0.005);  

      //float closestDepth = texture( shadow_tex , shadow_proj.xy).r;
      //float shadow = currentDepth - bias > closestDepth ? 0.0 : 1.0;  
      
      float shadow = 0;
      {
        vec2 texelSize = 1.0 / textureSize(shadow_tex, 0);
        for(int x = -1; x <= 1; ++x)
        {
          for(int y = -1; y <= 1; ++y)
          {
            float pcfDepth = texture(shadow_tex,  shadow_proj.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth ? 0.0 : 1.0;        
          }    
        }
        shadow /= 9.0;
      }

      float dot_prod = dot (direction_to_light_eye, normalize(normal_eye));
      float clamped_dot_prod = abs(max (dot_prod, -double_sided));
      vec3 Id = Ld * vec3(Kdi) * clamped_dot_prod;    // Diffuse intensity

      vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normal_eye));
      vec3 surface_to_viewer_eye = normalize (-position_eye);
      float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
      dot_prod_specular = float(abs(dot_prod)==dot_prod) * abs(max (dot_prod_specular, -double_sided));
      float specular_factor = pow (dot_prod_specular, specular_exponent);
      vec3 Is = Ls * vec3(Ksi) * specular_factor;    // specular intensity
      vec4 color = vec4(Ia + shadow*(lighting_factor * (Is + Id) + (1.0-lighting_factor) * vec3(Kdi)),(Kai.a+Ksi.a+Kdi.a)/3);
      outColor = mix(vec4(1,1,1,1), texture(tex, texcoordi), texture_factor) * color;
      if (fixed_color != vec4(0.0)) outColor = fixed_color;

    }
  }
)";
  igl::opengl::create_shader_program(
  mesh_vertex_shader_string,
  mesh_fragment_shader_string,
  {},
  vr.data().meshgl.shader_mesh);

  vr.callback_pre_draw = [&](decltype(vr) &)->bool
  {
    igl::opengl::report_gl_error("begin pre_draw");
    // attach buffers
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_depth_fbo);
    glBindRenderbuffer(GL_RENDERBUFFER, shadow_color_rbo);

    // clear buffer 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // In the libigl viewer setup, each mesh has its own shader program. This is
    // kind of funny because they should all be the same, just different uniform
    // values.

    glViewport(0,0,shadow_width,shadow_height);
    {
      double t = igl::get_seconds();
      vr.core().light_position = Eigen::Vector3f( cos(t), 0.5, sin(t)+0.5 );
    }
    Eigen::Vector3f camera_eye = vr.core().light_position.normalized()*5;
    Eigen::Vector3f camera_up = [&]()
      {
        Eigen::Matrix<float,3,2> T;
        igl::null(camera_eye.transpose().eval(),T);
        return T.col(0);
      }();
    Eigen::Vector3f camera_center = vr.core().camera_center;
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
    float camera_view_angle = vr.core().camera_view_angle;
    float camera_dnear = vr.core().camera_dnear;
    float camera_dfar = vr.core().camera_dfar;
    camera_dfar = exp2( 0.5 * ( log2(camera_dnear) + log2(camera_dfar)));
    igl::look_at( camera_eye, camera_center, camera_up, view);

    Eigen::Quaternionf trackball_angle = vr.core().trackball_angle;
    float camera_zoom = vr.core().camera_zoom;
    float camera_base_zoom = vr.core().camera_base_zoom;
    Eigen::Vector3f camera_translation = vr.core().camera_translation;
    Eigen::Vector3f camera_base_translation = vr.core().camera_base_translation;
    view = view
      * (trackball_angle * Eigen::Scaling(camera_zoom * camera_base_zoom)
      * Eigen::Translation3f(camera_translation + camera_base_translation)).matrix();

    float length = (camera_eye - camera_center).norm();
    float h = tan(camera_view_angle/360.0 * igl::PI) * (length);
    igl::ortho(-h*shadow_width/shadow_height, h*shadow_width/shadow_height, -h, h, camera_dnear, camera_dfar,proj);
    Eigen::Matrix4f norm = view.inverse().transpose();

    //// This is not working as expected.
    //glCullFace(GL_FRONT);
    //glEnable(GL_CULL_FACE);
    for(auto & data : vr.data_list)
    {
      if (data.dirty)
      {
        data.updateGL(data, data.invert_normals, data.meshgl);
        data.dirty = igl::opengl::MeshGL::DIRTY_NONE;
      }
      data.meshgl.bind_mesh();
      // Send transformations to the GPU
      GLint viewi  = glGetUniformLocation(data.meshgl.shader_mesh,"view");
      GLint proji  = glGetUniformLocation(data.meshgl.shader_mesh,"proj");
      GLint normi  = glGetUniformLocation(data.meshgl.shader_mesh,"normal_matrix");
      glUniformMatrix4fv(viewi, 1, GL_FALSE, view.data());
      glUniformMatrix4fv(glGetUniformLocation(data.meshgl.shader_mesh,"shadow_view"), 1, GL_FALSE, view.data());
      glUniformMatrix4fv(glGetUniformLocation(data.meshgl.shader_mesh,"shadow_proj"), 1, GL_FALSE, proj.data());
      glUniformMatrix4fv(proji, 1, GL_FALSE, proj.data());
      glUniformMatrix4fv(normi, 1, GL_FALSE, norm.data());
      // Light parameters
      GLint specular_exponenti    = glGetUniformLocation(data.meshgl.shader_mesh,"specular_exponent");
      GLint light_position_eyei = glGetUniformLocation(data.meshgl.shader_mesh,"light_position_eye");
      GLint lighting_factori      = glGetUniformLocation(data.meshgl.shader_mesh,"lighting_factor");
      GLint fixed_colori          = glGetUniformLocation(data.meshgl.shader_mesh,"fixed_color");
      GLint texture_factori       = glGetUniformLocation(data.meshgl.shader_mesh,"texture_factor");
      GLint matcap_factori        = glGetUniformLocation(data.meshgl.shader_mesh,"matcap_factor");
      GLint double_sidedi         = glGetUniformLocation(data.meshgl.shader_mesh,"double_sided");
      glUniform1f(specular_exponenti, data.shininess);
      glUniform3fv(light_position_eyei, 1, vr.core().light_position.data());
      glUniform1f(lighting_factori, vr.core().lighting_factor); // enables lighting
      glUniform4f(fixed_colori, 0.0, 0.0, 0.0, 0.0);
      // Texture
      glUniform1f(texture_factori, 0.0f);
      glUniform1f(matcap_factori, 0.0f);
      glUniform1f(double_sidedi, 1.0f);
      glUniform1i(glGetUniformLocation(data.meshgl.shader_mesh,"shadow_pass"),true);
      data.meshgl.draw_mesh(true);
      glUniform1i(glGetUniformLocation(data.meshgl.shader_mesh,"shadow_pass"),false);
    }
    //glDisable(GL_CULL_FACE);
    //glCullFace(GL_BACK);

    // Set texture before draw
    glActiveTexture(GL_TEXTURE0+1);
    glBindTexture(GL_TEXTURE_2D, shadow_depth_tex);
    igl::opengl::report_gl_error("after bind texture");
    for(auto & data : vr.data_list)
    {
      glUniform1i(glGetUniformLocation(data.meshgl.shader_mesh,"shadow_tex"), 1);
    }


    // set up matrices in shader
    //// draw meshes
    //{
    //  typedef unsigned char Scalar;
    //  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>  R,G,B,A,D;
    //  read_pixels(shadow_width,shadow_height,R,G,B,A,D);
    //  printf("R ∈ [%4d, %4d]\n",(int)R.minCoeff(),(int)R.maxCoeff());
    //  printf("G ∈ [%4d, %4d]\n",(int)G.minCoeff(),(int)G.maxCoeff());
    //  printf("B ∈ [%4d, %4d]\n",(int)B.minCoeff(),(int)B.maxCoeff());
    //  printf("A ∈ [%4d, %4d]\n",(int)A.minCoeff(),(int)A.maxCoeff());
    //  printf("D ∈ [%4d, %4d]\n",(int)D.minCoeff(),(int)D.maxCoeff());
    //  igl::png::writePNG(R,G,B,A,"rgba.png");
    //  igl::png::writePNG(D,D,D,A,"ddda.png");
    //}

    //{
    //  typedef double Scalar;
    //  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>  R,G,B,A,D;
    //  read_pixels(shadow_width,shadow_height,R,G,B,A,D);
    //  Eigen::Matrix<Scalar,Eigen::Dynamic,1> U;
    //  igl::unique(D,U);
    //  printf("|U| = %d\n",U.size());
    //}

    
    

    // unattach buffers
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    igl::opengl::report_gl_error("end pre_draw");
    return false;
  };

  vr.launch_rendering(true);
  vr.launch_shut();
}

#undef IGL_STATIC_LIBRARY
#include <igl/unique.cpp>
template void igl::unique<Eigen::Matrix<unsigned char, -1, -1, 0, -1, -1>, Eigen::Matrix<unsigned char, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<unsigned char, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<unsigned char, -1, 1, 0, -1, 1> >&);
