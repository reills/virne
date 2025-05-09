graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 734
  arrival_time 15383.912077771616
  lifetime 1210.806633855763
  num_nodes 2
  type "path"
  node [
    id 0
    label "0"
    cpu 26
    gpu 38
    rom 40
  ]
  node [
    id 1
    label "1"
    cpu 24
    gpu 48
    rom 18
  ]
  edge [
    source 0
    target 1
    bw 17
  ]
]
