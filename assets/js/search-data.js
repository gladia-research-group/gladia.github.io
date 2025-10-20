// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-",
    title: "",
    section: "Navigation",
    handler: () => {
      window.location.href = "/blog/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/index.html";
          },
        },{id: "post-beyond-attention-as-a-graph",
        
          title: "Beyond Attention as a Graph",
        
        description: "Higher-order (n-simplicial) attention as topology-driven message passing beyond graphs.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/beyond_attention_as_a_graph/";
          
        },
      },{id: "post-model-merging-a-biased-overview",
        
          title: "Model Merging — a biased overview",
        
        description: "A friendly tour of model merging, suspiciously aligned with my own research.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/model_merging/";
          
        },
      },{id: "post-attention-sinks-from-the-graph-perspective",
        
          title: "Attention sinks from the graph perspective",
        
        description: "Why causal transformers naturally concentrate attention on their earliest tokens.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/attention_sinks/";
          
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%6F%75@%65%78%61%6D%70%6C%65.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-inspire',
        title: 'Inspire HEP',
        section: 'Socials',
        handler: () => {
          window.open("https://inspirehep.net/authors/1010907", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/blog/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=qc6CJjYAAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.alberteinstein.com/", "_blank");
        },
      },];
