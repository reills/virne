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
  id 1765
  arrival_time 39427.059738703174
  lifetime 621.1736254170104
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 49
    gpu 13
    rom 28
  ]
  node [
    id 1
    label "1"
    cpu 5
    gpu 24
    rom 16
  ]
  node [
    id 2
    label "2"
    cpu 6
    gpu 36
    rom 41
  ]
  node [
    id 3
    label "3"
    cpu 46
    gpu 32
    rom 22
  ]
  node [
    id 4
    label "4"
    cpu 27
    gpu 20
    rom 22
  ]
  edge [
    source 0
    target 1
    bw 27
  ]
  edge [
    source 1
    target 2
    bw 32
  ]
  edge [
    source 2
    target 3
    bw 37
  ]
  edge [
    source 3
    target 4
    bw 15
  ]
]
