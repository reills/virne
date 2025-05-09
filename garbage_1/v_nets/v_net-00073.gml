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
  id 73
  arrival_time 1344.4060877530135
  lifetime 1961.1740786317407
  num_nodes 6
  type "path"
  node [
    id 0
    label "0"
    cpu 44
    gpu 3
    rom 4
  ]
  node [
    id 1
    label "1"
    cpu 10
    gpu 9
    rom 10
  ]
  node [
    id 2
    label "2"
    cpu 28
    gpu 29
    rom 20
  ]
  node [
    id 3
    label "3"
    cpu 31
    gpu 39
    rom 27
  ]
  node [
    id 4
    label "4"
    cpu 15
    gpu 1
    rom 30
  ]
  node [
    id 5
    label "5"
    cpu 32
    gpu 0
    rom 39
  ]
  edge [
    source 0
    target 1
    bw 46
  ]
  edge [
    source 1
    target 2
    bw 9
  ]
  edge [
    source 2
    target 3
    bw 10
  ]
  edge [
    source 3
    target 4
    bw 28
  ]
  edge [
    source 4
    target 5
    bw 41
  ]
]
