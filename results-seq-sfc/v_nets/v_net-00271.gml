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
  id 271
  arrival_time 5248.329447754198
  lifetime 2665.5676668018705
  num_nodes 6
  type "path"
  node [
    id 0
    label "0"
    cpu 39
    gpu 27
    rom 14
  ]
  node [
    id 1
    label "1"
    cpu 2
    gpu 10
    rom 38
  ]
  node [
    id 2
    label "2"
    cpu 1
    gpu 5
    rom 28
  ]
  node [
    id 3
    label "3"
    cpu 12
    gpu 28
    rom 45
  ]
  node [
    id 4
    label "4"
    cpu 12
    gpu 17
    rom 23
  ]
  node [
    id 5
    label "5"
    cpu 7
    gpu 11
    rom 24
  ]
  edge [
    source 0
    target 1
    bw 42
  ]
  edge [
    source 1
    target 2
    bw 31
  ]
  edge [
    source 2
    target 3
    bw 5
  ]
  edge [
    source 3
    target 4
    bw 3
  ]
  edge [
    source 4
    target 5
    bw 48
  ]
]
