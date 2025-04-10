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
  id 1083
  arrival_time 22631.54421101994
  lifetime 2.6551911692446706
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 16
    gpu 11
    rom 22
  ]
  node [
    id 1
    label "1"
    cpu 23
    gpu 14
    rom 37
  ]
  node [
    id 2
    label "2"
    cpu 8
    gpu 39
    rom 9
  ]
  node [
    id 3
    label "3"
    cpu 47
    gpu 27
    rom 37
  ]
  node [
    id 4
    label "4"
    cpu 41
    gpu 50
    rom 24
  ]
  edge [
    source 0
    target 1
    bw 40
  ]
  edge [
    source 1
    target 2
    bw 39
  ]
  edge [
    source 2
    target 3
    bw 31
  ]
  edge [
    source 3
    target 4
    bw 27
  ]
]
